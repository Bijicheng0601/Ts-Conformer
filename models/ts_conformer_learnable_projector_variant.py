"""Teacher-student augmented LOSO Conformer with a learnable feature projector.

This script extends the augmented LOSO pipeline with a three-timepoint teacher
and a two-timepoint student:

1. Teacher branch uses T0/T1/T2 augmented ERP samples during training.
2. Student branch uses T0/T1 augmented ERP samples during training and testing.
3. Distillation combines teacher supervision, student classification, KL-based
   soft target learning, and feature alignment.
4. Testing uses only the student branch with vote-based T0/T1 inference.
"""

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TS_DATA_ROOT", "restricted_data")) / 'ADHDtime_preprocessed'
LABEL_CSV = Path(os.environ.get("TS_DATA_ROOT", "restricted_data")) / 'ADHDtime/ant_treatment_response_by_subject_core2.csv'
BASE_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
LABEL_COL = "final_response_label_core2"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEACHER_INPUT_TIMEPOINTS = ["T0", "T1", "T2"]
STUDENT_INPUT_TIMEPOINTS = ["T0", "T1"]
ALPHA = 1.0
BETA = 0.5
GAMMA = 0.1
DISTILL_TEMPERATURE = 2.0
MAX_EPOCHS = 120
PATIENCE = 15

warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mne():
    import os

    mne_home = Path(__file__).resolve().parents[1] / "outputs"
    mne_home.mkdir(parents=True, exist_ok=True)
    os.environ["_MNE_FAKE_HOME_DIR"] = str(mne_home)
    os.environ["MNE_HOME"] = str(mne_home)
    os.environ["MNE_DONTWRITE_HOME"] = "true"
    import mne

    return mne


def find_epoch_file(subject: str, timepoint: str) -> Path:
    subject_dir = DATA_ROOT / subject
    candidates = list(subject_dir.glob(f"{timepoint}*_epo.fif")) + list(subject_dir.glob(f"{timepoint.lower()}*_epo.fif"))
    if not candidates:
        raise FileNotFoundError(f"missing {timepoint} epo file for {subject}")
    return sorted(candidates)[0]


@dataclass
class SubjectData:
    subject: str
    label: int
    t0_trials: np.ndarray
    t1_trials: np.ndarray
    t2_trials: np.ndarray
    t0_avg: np.ndarray
    t1_avg: np.ndarray
    t2_avg: np.ndarray


def load_subject_trials(subject: str, timepoint: str, mne) -> tuple[np.ndarray, np.ndarray]:
    epo_path = find_epoch_file(subject, timepoint)
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose="ERROR")
    data = epochs.get_data(copy=True).astype(np.float32)
    avg = data.mean(axis=0).astype(np.float32)
    return data, avg


def euclidean_align_trials(trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply subject-timepoint Euclidean Alignment to epochs before averaging."""
    cov = np.einsum("nct,ndt->ncd", trials, trials) / float(trials.shape[2])
    ref_cov = cov.mean(axis=0)
    ref_cov = (ref_cov + ref_cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(ref_cov)
    eigvals = np.clip(eigvals, eps, None)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    aligned = np.einsum("cd,ndt->nct", whitening.astype(np.float32), trials)
    return aligned.astype(np.float32)


def build_dataset(use_ea: bool = False, ea_eps: float = 1e-6) -> tuple[list[SubjectData], list[dict[str, str]]]:
    """Load T0/T1/T2 epochs and keep only subjects with complete three-timepoint data."""
    mne = load_mne()
    label_df = pd.read_csv(LABEL_CSV)[["subject", LABEL_COL]].sort_values("subject")
    subjects: list[SubjectData] = []
    skipped: list[dict[str, str]] = []
    for row in label_df.itertuples(index=False):
        subject = row.subject
        label = 1 if getattr(row, LABEL_COL) == "effective" else 0
        try:
            t0_trials, t0_avg = load_subject_trials(subject, "T0", mne)
            t1_trials, t1_avg = load_subject_trials(subject, "T1", mne)
            t2_trials, t2_avg = load_subject_trials(subject, "T2", mne)
        except FileNotFoundError as exc:
            skipped.append({"subject": subject, "reason": str(exc)})
            continue
        if use_ea:
            t0_trials = euclidean_align_trials(t0_trials, eps=ea_eps)
            t1_trials = euclidean_align_trials(t1_trials, eps=ea_eps)
            t2_trials = euclidean_align_trials(t2_trials, eps=ea_eps)
            t0_avg = t0_trials.mean(axis=0).astype(np.float32)
            t1_avg = t1_trials.mean(axis=0).astype(np.float32)
            t2_avg = t2_trials.mean(axis=0).astype(np.float32)
        subjects.append(
            SubjectData(
                subject=subject,
                label=label,
                t0_trials=t0_trials,
                t1_trials=t1_trials,
                t2_trials=t2_trials,
                t0_avg=t0_avg,
                t1_avg=t1_avg,
                t2_avg=t2_avg,
            )
        )
    return sorted(subjects, key=lambda s: s.subject), skipped


class ERPBranch(nn.Module):
    """Shared ERP encoder used by teacher and student branches."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        emb_dim: int = 32,
        n_heads: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel_size),
                padding=(0, temporal_padding),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(temporal_filters, emb_dim, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.ELU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=n_heads,
                dim_feedforward=emb_dim * 4,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = x.squeeze(2).transpose(1, 2)
        if self.use_attention:
            x = self.transformer(x)
        x = x.mean(dim=1)
        return self.dropout(x)


class TeacherLongitudinalConformer3T(nn.Module):
    """Three-timepoint teacher that uses T0/T1/T2 embeddings and delta features."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        emb_dim: int,
        use_attention: bool = True,
    ):
        super().__init__()
        self.branch = ERPBranch(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            emb_dim=emb_dim,
            use_attention=use_attention,
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 6, 96),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(96, 1),
        )

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = self.branch(x_t0)
        z1 = self.branch(x_t1)
        z2 = self.branch(x_t2)
        z10 = z1 - z0
        z21 = z2 - z1
        z20 = z2 - z0
        joint = torch.cat([z0, z1, z2, z10, z21, z20], dim=1)
        logits = self.classifier(joint).squeeze(1)
        return logits, joint


class StudentLongitudinalConformer2T(nn.Module):
    """Two-timepoint student used for both training and deployment."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        emb_dim: int,
        use_attention: bool = True,
    ):
        super().__init__()
        self.branch = ERPBranch(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            emb_dim=emb_dim,
            use_attention=use_attention,
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = self.branch(x_t0)
        z1 = self.branch(x_t1)
        z10 = z1 - z0
        joint = torch.cat([z0, z1, z10], dim=1)
        logits = self.classifier(joint).squeeze(1)
        return logits, joint


class TeacherStudentConformer(nn.Module):
    """Wrapper containing teacher, student and a projection head for feature distillation."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        emb_dim: int,
        use_attention: bool = True,
    ):
        super().__init__()
        self.teacher = TeacherLongitudinalConformer3T(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            emb_dim=emb_dim,
            use_attention=use_attention,
        )
        self.student = StudentLongitudinalConformer2T(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            emb_dim=emb_dim,
            use_attention=use_attention,
        )
        self.teacher_projector = nn.Linear(emb_dim * 6, emb_dim * 3)

    def forward_teacher_student(
        self,
        x_t0: torch.Tensor,
        x_t1: torch.Tensor,
        x_t2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_logits, teacher_joint = self.teacher(x_t0, x_t1, x_t2)
        student_logits, student_joint = self.student(x_t0, x_t1)
        # P remains learnable, while feature loss cannot update the teacher.
        teacher_proj = self.teacher_projector(teacher_joint.detach())
        return teacher_logits, teacher_proj, student_logits, student_joint

    def forward_student(self, x_t0: torch.Tensor, x_t1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student(x_t0, x_t1)


def make_augmented_triplets(
    subjects: list[SubjectData],
    n_aug_per_subject: int,
    n_trials_per_avg: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build aligned T0/T1/T2 augmented samples for teacher-student training."""
    rng = np.random.default_rng(seed)
    x0_list = []
    x1_list = []
    x2_list = []
    y_list = []
    for subj in subjects:
        n0 = subj.t0_trials.shape[0]
        n1 = subj.t1_trials.shape[0]
        n2 = subj.t2_trials.shape[0]
        for _ in range(n_aug_per_subject):
            idx0 = rng.choice(n0, size=min(n_trials_per_avg, n0), replace=n0 < n_trials_per_avg)
            idx1 = rng.choice(n1, size=min(n_trials_per_avg, n1), replace=n1 < n_trials_per_avg)
            idx2 = rng.choice(n2, size=min(n_trials_per_avg, n2), replace=n2 < n_trials_per_avg)
            x0_list.append(subj.t0_trials[idx0].mean(axis=0))
            x1_list.append(subj.t1_trials[idx1].mean(axis=0))
            x2_list.append(subj.t2_trials[idx2].mean(axis=0))
            y_list.append(subj.label)
    x0 = torch.tensor(np.stack(x0_list), dtype=torch.float32)
    x1 = torch.tensor(np.stack(x1_list), dtype=torch.float32)
    x2 = torch.tensor(np.stack(x2_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    return x0, x1, x2, y


def make_augmented_pairs_for_subject(
    subject: SubjectData,
    n_pairs: int,
    n_trials_per_avg: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Build vote samples for one held-out subject during testing using only T0/T1."""
    rng = np.random.default_rng(seed)
    x0_list = []
    x1_list = []
    vote_index_rows = []
    n0 = subject.t0_trials.shape[0]
    n1 = subject.t1_trials.shape[0]
    for vote_idx in range(1, n_pairs + 1):
        idx0 = rng.choice(n0, size=min(n_trials_per_avg, n0), replace=n0 < n_trials_per_avg)
        idx1 = rng.choice(n1, size=min(n_trials_per_avg, n1), replace=n1 < n_trials_per_avg)
        x0_list.append(subject.t0_trials[idx0].mean(axis=0))
        x1_list.append(subject.t1_trials[idx1].mean(axis=0))
        vote_index_rows.append(
            {
                "vote_index": vote_idx,
                "t0_trial_indices": ";".join(map(str, idx0.tolist())),
                "t1_trial_indices": ";".join(map(str, idx1.tolist())),
            }
        )
    x0 = torch.tensor(np.stack(x0_list), dtype=torch.float32)
    x1 = torch.tensor(np.stack(x1_list), dtype=torch.float32)
    return x0, x1, vote_index_rows


def make_average_triplets(subjects: list[SubjectData]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = torch.tensor(np.stack([s.t0_avg for s in subjects]), dtype=torch.float32)
    x1 = torch.tensor(np.stack([s.t1_avg for s in subjects]), dtype=torch.float32)
    x2 = torch.tensor(np.stack([s.t2_avg for s in subjects]), dtype=torch.float32)
    y = torch.tensor([s.label for s in subjects], dtype=torch.float32)
    return x0, x1, x2, y


def add_gaussian_noise(
    x: torch.Tensor,
    noise_std: float,
    seed: int,
    noisy_channels_per_sample: int = 10,
) -> torch.Tensor:
    if noise_std <= 0:
        return x
    device = x.device
    generator = torch.Generator(device=device if x.is_cuda else "cpu")
    generator.manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=device) * noise_std
    mask = torch.zeros_like(x)
    n_samples, n_channels, _ = x.shape
    n_select = min(noisy_channels_per_sample, n_channels)
    for sample_idx in range(n_samples):
        selected = torch.randperm(n_channels, generator=generator, device=device)[:n_select]
        mask[sample_idx, selected, :] = 1.0
    return x + noise * mask


def zscore_by_train_triplets(
    train_x0: torch.Tensor,
    train_x1: torch.Tensor,
    train_x2: torch.Tensor,
    other_x0: torch.Tensor,
    other_x1: torch.Tensor,
    other_x2: torch.Tensor | None = None,
):
    concat = torch.cat([train_x0, train_x1, train_x2], dim=0)
    mean = concat.mean(dim=(0, 2), keepdim=True)
    std = concat.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    train_x0 = (train_x0 - mean) / std
    train_x1 = (train_x1 - mean) / std
    train_x2 = (train_x2 - mean) / std
    other_x0 = (other_x0 - mean) / std
    other_x1 = (other_x1 - mean) / std
    if other_x2 is not None:
        other_x2 = (other_x2 - mean) / std
    return train_x0, train_x1, train_x2, other_x0, other_x1, other_x2


def select_val_subjects(
    train_subjects: list[SubjectData],
    fixed_val_effective: int | None = None,
    fixed_val_ineffective: int | None = None,
) -> set[str]:
    subjects = [s.subject for s in train_subjects]
    labels = [s.label for s in train_subjects]
    if fixed_val_effective is not None and fixed_val_ineffective is not None:
        effective_subjects = sorted([s.subject for s in train_subjects if s.label == 1])
        ineffective_subjects = sorted([s.subject for s in train_subjects if s.label == 0])
        n_eff = min(fixed_val_effective, len(effective_subjects))
        n_ineff = min(fixed_val_ineffective, len(ineffective_subjects))
        return set(effective_subjects[:n_eff] + ineffective_subjects[:n_ineff])
    if len(set(labels)) < 2 or len(subjects) < 6:
        n_val = max(2, int(round(len(subjects) * 0.2)))
        return set(subjects[:n_val])
    _, sub_val = train_test_split(
        subjects,
        test_size=max(0.2, 4 / len(subjects)),
        random_state=SEED,
        stratify=labels,
    )
    return set(sub_val)


def soft_binary_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Binary KL over softened probabilities represented as two-class distributions."""
    teacher_prob = torch.sigmoid(teacher_logits / temperature)
    student_prob = torch.sigmoid(student_logits / temperature)
    teacher_dist = torch.stack([1.0 - teacher_prob, teacher_prob], dim=1).clamp(1e-6, 1 - 1e-6)
    student_dist = torch.stack([1.0 - student_prob, student_prob], dim=1).clamp(1e-6, 1 - 1e-6)
    loss = F.kl_div(student_dist.log(), teacher_dist, reduction="batchmean")
    return loss * (temperature**2)


def plot_fold_visualization(
    fold_idx: int,
    subject: str,
    true_label: int,
    subject_prob: float,
    vote_probs: np.ndarray,
    output_dir: Path,
    decision_threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    true_label_name = "effective" if true_label == 1 else "ineffective"
    pred_label_name = "effective" if subject_prob >= decision_threshold else "ineffective"
    color = "#2ca02c" if true_label == 1 else "#d62728"

    axes[0].bar(["subject"], [subject_prob], color=color, alpha=0.85, width=0.5)
    axes[0].axhline(decision_threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Predicted P(effective)")
    axes[0].set_title(
        f"Fold {fold_idx:02d} Subject-Level Result: {subject}\n"
        f"true={true_label_name}, pred={pred_label_name}, "
        f"threshold={decision_threshold:.2f}, mean_prob={subject_prob:.3f}"
    )
    axes[0].text(
        0,
        min(subject_prob + 0.04, 0.96),
        f"{subject_prob:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    vote_idx = np.arange(1, len(vote_probs) + 1)
    axes[1].plot(vote_idx, vote_probs, color=color, linewidth=1.5, alpha=0.85)
    axes[1].scatter(vote_idx, vote_probs, color=color, s=16, alpha=0.85)
    axes[1].axhline(decision_threshold, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(subject_prob, color="#1f77b4", linestyle="-.", linewidth=1.2)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Vote Index")
    axes[1].set_ylabel("Vote-Level P(effective)")
    axes[1].set_title(
        f"Fold {fold_idx:02d} Trial-Level Votes ({len(vote_probs)} votes, "
        f"mean={subject_prob:.3f}, std={vote_probs.std(ddof=0):.3f})"
    )

    fig.savefig(
        output_dir / f"fold_{fold_idx:02d}_{subject}_subject_and_trial_level.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_summary_visualizations(
    pred_df: pd.DataFrame,
    vote_df: pd.DataFrame,
    output_dir: Path,
    decision_threshold: float,
) -> None:
    subject_plot = output_dir / "all_folds_subject_level_summary.png"
    trial_plot = output_dir / "all_folds_trial_level_summary.png"

    plot_df = pred_df.copy().sort_values("predicted_probability_effective", ascending=False).reset_index(drop=True)
    colors = plot_df["true_label"].map({1: "#2ca02c", 0: "#d62728"})

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(plot_df["subject"], plot_df["predicted_probability_effective"], color=colors, alpha=0.85)
    ax.axhline(decision_threshold, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted P(effective)")
    ax.set_xlabel("Subject")
    ax.set_title(f"Subject-Level LOSO Predictions Across All Folds (threshold={decision_threshold:.2f})")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.savefig(subject_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    ordered_subjects = plot_df["subject"].tolist()
    positions = np.arange(len(ordered_subjects))
    subject_to_pos = {subj: pos for pos, subj in enumerate(ordered_subjects)}

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    for label_value, color, label_name in [(1, "#2ca02c", "true effective"), (0, "#d62728", "true ineffective")]:
        sub_df = vote_df[vote_df["true_label"] == label_value]
        x = sub_df["subject"].map(subject_to_pos).astype(float).to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(sub_df))
        ax.scatter(x + jitter, sub_df["vote_probability_effective"], s=14, alpha=0.35, color=color, label=label_name)
    ax.axhline(decision_threshold, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered_subjects, rotation=60, ha="right")
    ax.set_ylabel("Vote-Level P(effective)")
    ax.set_xlabel("Subject")
    ax.set_title(f"Trial-Level Vote Distributions Across All LOSO Folds (threshold={decision_threshold:.2f})")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False)
    fig.savefig(trial_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    fold_idx: int,
    subject: str,
    history_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(history_df["epoch"], history_df["teacher_train_loss"], label="Teacher Train Loss", linewidth=1.8, color="#1f77b4")
    ax.plot(history_df["epoch"], history_df["student_train_loss"], label="Student Train Loss", linewidth=1.8, color="#ff7f0e")
    ax.plot(history_df["epoch"], history_df["student_val_loss"], label="Student Val Loss", linewidth=1.8, color="#d62728")
    best_idx = history_df["student_val_loss"].idxmin()
    best_epoch = int(history_df.loc[best_idx, "epoch"])
    best_val = float(history_df.loc[best_idx, "student_val_loss"])
    ax.scatter([best_epoch], [best_val], color="black", s=30, zorder=3, label=f"Best Student Val (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Fold {fold_idx:02d} Teacher-Student Training Curve: {subject}")
    ax.legend(frameon=False)
    fig.savefig(output_dir / f"fold_{fold_idx:02d}_{subject}_training_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_one_fold(
    train_subjects: list[SubjectData],
    val_subjects: set[str],
    test_subject: SubjectData,
    temporal_filters: int,
    temporal_kernel_size: int,
    temporal_padding: int,
    emb_dim: int,
    n_aug_per_subject: int,
    train_trials_per_avg: int,
    n_test_votes: int,
    test_trials_per_avg: int,
    train_noise_std: float,
    noisy_channels_per_sample: int,
    use_attention: bool,
    seed: int,
) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Train one LOSO fold and return student-based prediction outputs."""
    train_real = [s for s in train_subjects if s.subject not in val_subjects]
    val_real = [s for s in train_subjects if s.subject in val_subjects]

    train_x0, train_x1, train_x2, train_y = make_augmented_triplets(
        train_real,
        n_aug_per_subject=n_aug_per_subject,
        n_trials_per_avg=train_trials_per_avg,
        seed=seed,
    )
    val_x0, val_x1, val_x2, val_y = make_average_triplets(val_real)
    test_x0, test_x1, vote_index_rows = make_augmented_pairs_for_subject(
        test_subject,
        n_pairs=n_test_votes,
        n_trials_per_avg=test_trials_per_avg,
        seed=seed + 10000,
    )

    train_x0, train_x1, train_x2, val_x0, val_x1, val_x2 = zscore_by_train_triplets(
        train_x0, train_x1, train_x2, val_x0, val_x1, val_x2
    )
    train_x0, train_x1, train_x2, test_x0, test_x1, _ = zscore_by_train_triplets(
        train_x0, train_x1, train_x2, test_x0, test_x1, None
    )
    vote_debug_df = pd.DataFrame(vote_index_rows)
    vote_debug_df["t0_mean_abs_diff_from_vote1"] = (
        (test_x0 - test_x0[0:1]).abs().mean(dim=(1, 2)).detach().cpu().numpy()
    )
    vote_debug_df["t1_mean_abs_diff_from_vote1"] = (
        (test_x1 - test_x1[0:1]).abs().mean(dim=(1, 2)).detach().cpu().numpy()
    )
    vote_debug_df["paired_mean_abs_diff_from_vote1"] = (
        (
            (test_x0 - test_x0[0:1]).abs().mean(dim=(1, 2))
            + (test_x1 - test_x1[0:1]).abs().mean(dim=(1, 2))
        )
        / 2.0
    ).detach().cpu().numpy()

    train_x0 = add_gaussian_noise(
        train_x0,
        noise_std=train_noise_std,
        seed=seed + 20000,
        noisy_channels_per_sample=noisy_channels_per_sample,
    )
    train_x1 = add_gaussian_noise(
        train_x1,
        noise_std=train_noise_std,
        seed=seed + 30000,
        noisy_channels_per_sample=noisy_channels_per_sample,
    )
    train_x2 = add_gaussian_noise(
        train_x2,
        noise_std=train_noise_std,
        seed=seed + 40000,
        noisy_channels_per_sample=noisy_channels_per_sample,
    )

    model = TeacherStudentConformer(
        n_channels=train_x0.shape[1],
        temporal_filters=temporal_filters,
        temporal_kernel_size=temporal_kernel_size,
        temporal_padding=temporal_padding,
        emb_dim=emb_dim,
        use_attention=use_attention,
    ).to(DEVICE)

    pos_count = float(train_y.sum().item())
    neg_count = float(len(train_y) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    train_x0 = train_x0.to(DEVICE)
    train_x1 = train_x1.to(DEVICE)
    train_x2 = train_x2.to(DEVICE)
    train_y = train_y.to(DEVICE)
    val_x0 = val_x0.to(DEVICE)
    val_x1 = val_x1.to(DEVICE)
    val_y = val_y.to(DEVICE)
    test_x0 = test_x0.to(DEVICE)
    test_x1 = test_x1.to(DEVICE)

    best_state = None
    best_val = math.inf
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        teacher_logits, teacher_proj, student_logits, student_joint = model.forward_teacher_student(train_x0, train_x1, train_x2)
        teacher_cls = criterion(teacher_logits, train_y)
        student_cls = criterion(student_logits, train_y)
        kd_loss = soft_binary_kl(student_logits, teacher_logits.detach(), DISTILL_TEMPERATURE)
        feat_loss = F.mse_loss(student_joint, teacher_proj)
        total_loss = teacher_cls + ALPHA * student_cls + BETA * kd_loss + GAMMA * feat_loss

        total_loss.backward()
        projector_grad = model.teacher_projector.weight.grad
        if projector_grad is None or not torch.isfinite(projector_grad).all():
            raise RuntimeError("Learnable projector did not receive a finite gradient.")
        projector_grad_norm = float(projector_grad.norm().item())
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_student_logits, _ = model.forward_student(val_x0, val_x1)
            val_student_loss = criterion(val_student_logits, val_y).item()

        history_rows.append(
            {
                "epoch": epoch,
                "teacher_train_loss": float(teacher_cls.item()),
                "student_train_loss": float(student_cls.item()),
                "distill_loss": float(kd_loss.item()),
                "feature_loss": float(feat_loss.item()),
                "projector_grad_norm": projector_grad_norm,
                "total_train_loss": float(total_loss.item()),
                "student_val_loss": float(val_student_loss),
            }
        )

        if val_student_loss < best_val:
            best_val = val_student_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        vote_logits, _ = model.forward_student(test_x0, test_x1)
        vote_probs = torch.sigmoid(vote_logits).detach().cpu().numpy().astype(np.float32)
        prob = float(vote_probs.mean())

    history_df = pd.DataFrame(history_rows)
    return prob, vote_probs, history_df, vote_debug_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-filters", type=int, default=16)
    parser.add_argument("--temporal-kernel-size", type=int, default=25)
    parser.add_argument("--temporal-padding", type=int, default=12)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--n-aug-per-subject", type=int, default=32)
    parser.add_argument("--train-trials-per-avg", type=int, default=8)
    parser.add_argument("--n-test-votes", type=int, default=32)
    parser.add_argument("--test-trials-per-avg", type=int, default=8)
    parser.add_argument("--train-noise-std", type=float, default=0.0)
    parser.add_argument("--noisy-channels-per-sample", type=int, default=10)
    parser.add_argument("--use-ea", action="store_true", help="Apply subject-timepoint Euclidean Alignment before averaging.")
    parser.add_argument("--ea-eps", type=float, default=1e-6, help="Eigenvalue floor used by Euclidean Alignment.")
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Ablation: remove Transformer self-attention and use temporal mean pooling after convolution.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.55,
        help="Subject-level P(effective) threshold used to assign effective/ineffective labels.",
    )
    parser.add_argument("--fixed-val-effective", type=int, default=None, help="Use a fixed number of effective subjects in validation.")
    parser.add_argument("--fixed-val-ineffective", type=int, default=None, help="Use a fixed number of ineffective subjects in validation.")
    args = parser.parse_args()

    set_seed(SEED)
    output_dir = BASE_OUTPUT_DIR / (
        f"tf_{args.temporal_filters}_k{args.temporal_kernel_size}_"
        f"p{args.temporal_padding}_e{args.emb_dim}_a{args.n_aug_per_subject}_"
        f"tt{args.train_trials_per_avg}_v{args.n_test_votes}_tv{args.test_trials_per_avg}_"
        f"n{str(args.train_noise_std).replace('.', 'p')}"
        f"{'_ea' if args.use_ea else ''}"
        f"{'_noattn' if args.no_attention else ''}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training_curves"
    training_dir.mkdir(parents=True, exist_ok=True)
    fold_vis_dir = output_dir / "fold_visualizations"
    fold_vis_dir.mkdir(parents=True, exist_ok=True)

    subjects, skipped_subjects = build_dataset(use_ea=args.use_ea, ea_eps=args.ea_eps)
    rows = []
    vote_rows = []
    probs = []
    y_true = []

    print(
        f"Running teacher-student augmented LOSO Conformer | subjects={len(subjects)} | "
        f"filters={args.temporal_filters} | kernel={args.temporal_kernel_size} | "
        f"padding={args.temporal_padding} | emb_dim={args.emb_dim} | "
        f"aug={args.n_aug_per_subject} | train_trials_per_avg={args.train_trials_per_avg} | "
        f"test_votes={args.n_test_votes} | test_trials_per_avg={args.test_trials_per_avg} | "
        f"train_noise_std={args.train_noise_std} | noisy_channels={args.noisy_channels_per_sample} | "
        f"use_ea={args.use_ea} | ea_eps={args.ea_eps} | "
        f"use_attention={not args.no_attention} | "
        f"decision_threshold={args.decision_threshold} | "
        f"fixed_val_effective={args.fixed_val_effective} | "
        f"fixed_val_ineffective={args.fixed_val_ineffective}"
    )
    if skipped_subjects:
        print(f"Skipped {len(skipped_subjects)} subjects without complete T0/T1/T2 data.")

    progress = tqdm(
        enumerate(subjects, start=1),
        total=len(subjects),
        desc="LOSO folds",
        unit="fold",
    )

    for i, test_subject in progress:
        train_subjects = [s for s in subjects if s.subject != test_subject.subject]
        val_subjects = select_val_subjects(
            train_subjects,
            fixed_val_effective=args.fixed_val_effective,
            fixed_val_ineffective=args.fixed_val_ineffective,
        )
        print(
            f"[Fold {i:02d}/{len(subjects):02d}] test_subject={test_subject.subject} "
            f"true_label={'effective' if test_subject.label == 1 else 'ineffective'} | "
            f"train_subjects={len(train_subjects)} | val_subjects={len(val_subjects)}"
        )

        prob, vote_probs, history_df, vote_debug_df = train_one_fold(
            train_subjects,
            val_subjects,
            test_subject,
            temporal_filters=args.temporal_filters,
            temporal_kernel_size=args.temporal_kernel_size,
            temporal_padding=args.temporal_padding,
            emb_dim=args.emb_dim,
            n_aug_per_subject=args.n_aug_per_subject,
            train_trials_per_avg=args.train_trials_per_avg,
            n_test_votes=args.n_test_votes,
            test_trials_per_avg=args.test_trials_per_avg,
            train_noise_std=args.train_noise_std,
            noisy_channels_per_sample=args.noisy_channels_per_sample,
            use_attention=not args.no_attention,
            seed=SEED + i,
        )
        history_df.to_csv(
            training_dir / f"fold_{i:02d}_{test_subject.subject}_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_training_curve(i, test_subject.subject, history_df, training_dir)

        pred = 1 if prob >= args.decision_threshold else 0
        best_epoch = int(history_df.loc[history_df["student_val_loss"].idxmin(), "epoch"])
        best_val = float(history_df["student_val_loss"].min())
        print(
            f"[Fold {i:02d}/{len(subjects):02d}] pred_label={'effective' if pred == 1 else 'ineffective'} "
            f"prob={prob:.4f} | best_epoch={best_epoch} | best_student_val_loss={best_val:.4f}"
        )
        print(
            f"[Fold {i:02d}/{len(subjects):02d}] vote_probs="
            f"{np.array2string(vote_probs, precision=4, separator=', ')}"
        )
        print(
            f"[Fold {i:02d}/{len(subjects):02d}] vote_input_mean_abs_diff="
            f"{vote_debug_df['paired_mean_abs_diff_from_vote1'].mean():.6f} | "
            f"vote_prob_std={vote_probs.std(ddof=0):.6f}"
        )

        plot_fold_visualization(
            fold_idx=i,
            subject=test_subject.subject,
            true_label=test_subject.label,
            subject_prob=prob,
            vote_probs=vote_probs,
            output_dir=fold_vis_dir,
            decision_threshold=args.decision_threshold,
        )
        progress.set_postfix(
            subject=test_subject.subject,
            prob=f"{prob:.3f}",
            pred="eff" if pred == 1 else "ineff",
        )

        rows.append(
            {
                "fold": i,
                "subject": test_subject.subject,
                LABEL_COL: "effective" if test_subject.label == 1 else "ineffective",
                "true_label": int(test_subject.label),
                "predicted_probability_effective": prob,
                "true_label_probability": prob if test_subject.label == 1 else (1.0 - prob),
                "predicted_label": "effective" if pred == 1 else "ineffective",
            }
        )
        vote_debug_by_idx = vote_debug_df.set_index("vote_index")
        for vote_idx, vote_prob in enumerate(vote_probs, start=1):
            debug_row = vote_debug_by_idx.loc[vote_idx]
            vote_rows.append(
                {
                    "fold": i,
                    "subject": test_subject.subject,
                    "true_label": int(test_subject.label),
                    "vote_index": vote_idx,
                    "vote_probability_effective": float(vote_prob),
                    "t0_trial_indices": debug_row["t0_trial_indices"],
                    "t1_trial_indices": debug_row["t1_trial_indices"],
                    "t0_mean_abs_diff_from_vote1": float(debug_row["t0_mean_abs_diff_from_vote1"]),
                    "t1_mean_abs_diff_from_vote1": float(debug_row["t1_mean_abs_diff_from_vote1"]),
                    "paired_mean_abs_diff_from_vote1": float(debug_row["paired_mean_abs_diff_from_vote1"]),
                }
            )
        probs.append(prob)
        y_true.append(test_subject.label)

    pred_df = pd.DataFrame(rows)
    vote_df = pd.DataFrame(vote_rows)
    y_true_arr = np.array(y_true, dtype=int)
    prob_arr = np.array(probs, dtype=float)
    y_pred_arr = (np.array(probs) >= args.decision_threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision_by_class, recall_by_class, f1_by_class, support_by_class = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        zero_division=0,
    )
    true_counts = {
        "ineffective": int((y_true_arr == 0).sum()),
        "effective": int((y_true_arr == 1).sum()),
    }
    prediction_counts = {
        "ineffective": int((y_pred_arr == 0).sum()),
        "effective": int((y_pred_arr == 1).sum()),
    }
    metrics = {
        "n_subjects": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, prob_arr)),
        "pr_auc_effective": float(average_precision_score(y_true_arr, prob_arr)),
        "true_counts": true_counts,
        "prediction_counts": prediction_counts,
        "confusion_matrix_labels": ["ineffective", "effective"],
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_terms": {
            "true_ineffective_pred_ineffective_tn": int(tn),
            "true_ineffective_pred_effective_fp": int(fp),
            "true_effective_pred_ineffective_fn": int(fn),
            "true_effective_pred_effective_tp": int(tp),
        },
        "ineffective_precision": float(precision_by_class[0]),
        "ineffective_recall": float(recall_by_class[0]),
        "ineffective_f1": float(f1_by_class[0]),
        "ineffective_support": int(support_by_class[0]),
        "effective_precision": float(precision_by_class[1]),
        "effective_recall": float(recall_by_class[1]),
        "effective_f1": float(f1_by_class[1]),
        "effective_support": int(support_by_class[1]),
        "effective_prediction_rate": float(prediction_counts["effective"] / max(len(y_pred_arr), 1)),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["ineffective", "effective"],
            zero_division=0,
            output_dict=True,
        ),
        "device": DEVICE,
        "teacher_student_mode": True,
        "teacher_projector_learnable": True,
        "feature_gradient_path": "P(stopgrad(h_T))",
        "teacher_input_timepoints": TEACHER_INPUT_TIMEPOINTS,
        "student_input_timepoints": STUDENT_INPUT_TIMEPOINTS,
        "decision_threshold": args.decision_threshold,
        "input_type": "teacher uses T0/T1/T2 augmented ERP samples during training; student uses T0/T1 during training/testing",
        "architecture": "teacher-student shared-branch lightweight conformer",
        "use_attention": not args.no_attention,
        "attention_ablation": args.no_attention,
        "temporal_filters": args.temporal_filters,
        "temporal_kernel_size": args.temporal_kernel_size,
        "temporal_padding": args.temporal_padding,
        "emb_dim": args.emb_dim,
        "train_noise_std": args.train_noise_std,
        "noisy_channels_per_sample": args.noisy_channels_per_sample,
        "use_ea": args.use_ea,
        "ea_mode": "subject_timepoint" if args.use_ea else "none",
        "ea_eps": args.ea_eps,
        "distillation_temperature": DISTILL_TEMPERATURE,
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": GAMMA,
        "augmentation": {
            "n_aug_per_subject": args.n_aug_per_subject,
            "train_trials_per_average": args.train_trials_per_avg,
            "n_test_votes": args.n_test_votes,
            "test_trials_per_average": args.test_trials_per_avg,
            "test_aggregation": "mean_probability",
        },
        "skipped_subjects_missing_t2": skipped_subjects,
    }

    pred_df.to_csv(output_dir / "loso_predictions.csv", index=False, encoding="utf-8-sig")
    vote_df.to_csv(output_dir / "trial_level_vote_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "n_subjects": metrics["n_subjects"],
                "decision_threshold": metrics["decision_threshold"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc_effective": metrics["pr_auc_effective"],
                "true_ineffective": true_counts["ineffective"],
                "true_effective": true_counts["effective"],
                "pred_ineffective": prediction_counts["ineffective"],
                "pred_effective": prediction_counts["effective"],
                "effective_prediction_rate": metrics["effective_prediction_rate"],
                "ineffective_recall": metrics["ineffective_recall"],
                "effective_recall": metrics["effective_recall"],
                "ineffective_precision": metrics["ineffective_precision"],
                "effective_precision": metrics["effective_precision"],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        ]
    ).to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "model_info.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "label_csv": str(LABEL_CSV),
                "data_root": str(DATA_ROOT),
                "device": DEVICE,
                "teacher_student_mode": True,
                "teacher_projector_learnable": True,
                "feature_gradient_path": "P(stopgrad(h_T))",
                "teacher_input_timepoints": TEACHER_INPUT_TIMEPOINTS,
                "student_input_timepoints": STUDENT_INPUT_TIMEPOINTS,
                "decision_threshold": args.decision_threshold,
                "use_attention": not args.no_attention,
                "attention_ablation": args.no_attention,
                "temporal_filters": args.temporal_filters,
                "temporal_kernel_size": args.temporal_kernel_size,
                "temporal_padding": args.temporal_padding,
                "emb_dim": args.emb_dim,
                "train_noise_std": args.train_noise_std,
                "noisy_channels_per_sample": args.noisy_channels_per_sample,
                "use_ea": args.use_ea,
                "ea_mode": "subject_timepoint" if args.use_ea else "none",
                "ea_eps": args.ea_eps,
                "fixed_val_effective": args.fixed_val_effective,
                "fixed_val_ineffective": args.fixed_val_ineffective,
                "n_aug_per_subject": args.n_aug_per_subject,
                "train_trials_per_average": args.train_trials_per_avg,
                "n_test_votes": args.n_test_votes,
                "test_trials_per_average": args.test_trials_per_avg,
                "distillation_temperature": DISTILL_TEMPERATURE,
                "alpha": ALPHA,
                "beta": BETA,
                "gamma": GAMMA,
                "skipped_subjects_missing_t2": skipped_subjects,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_summary_visualizations(pred_df, vote_df, output_dir, decision_threshold=args.decision_threshold)

    print("saved:", output_dir / "loso_predictions.csv")
    print("saved:", output_dir / "trial_level_vote_predictions.csv")
    print("saved:", output_dir / "metrics_summary.csv")
    print("saved:", output_dir / "metrics.json")
    print("saved:", output_dir / "model_info.json")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
