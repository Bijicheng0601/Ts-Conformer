"""Teacher-student augmented LOSO ShallowConvNet for core2 response classification.

This script mirrors the teacher-student Conformer/EEGNet experiments but uses
a compact ShallowConvNet-style branch. It is intended as a lightweight baseline:

1. The teacher uses T0/T1/T2 during training.
2. The student uses T0/T1 during training and testing.
3. The same BCE, soft distillation, and feature-alignment losses are used.
4. Outputs are compatible with the existing Conformer/EEGNet result folders.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
from pathlib import Path

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
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.augmented_loso_teacher_student_conformer_core2 import (  # noqa: E402
    ALPHA,
    BETA,
    DATA_ROOT,
    DEVICE,
    DISTILL_TEMPERATURE,
    GAMMA,
    LABEL_COL,
    LABEL_CSV,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    STUDENT_INPUT_TIMEPOINTS,
    TEACHER_INPUT_TIMEPOINTS,
    add_gaussian_noise,
    build_dataset,
    make_augmented_pairs_for_subject,
    make_augmented_triplets,
    make_average_triplets,
    plot_fold_visualization,
    plot_summary_visualizations,
    plot_training_curve,
    select_val_subjects,
    set_seed,
    soft_binary_kl,
    zscore_by_train_triplets,
)


BASE_OUTPUT_DIR = PROJECT_ROOT / "augmented_loso_teacher_student_shallowconvnet_core2"


class ShallowConvNetBranch(nn.Module):
    """ShallowConvNet encoder for one ERP timepoint.

    The branch follows the classic temporal convolution -> spatial convolution
    -> square -> average-pool -> log pipeline, then projects to an embedding.
    """

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        spatial_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        pool_size: int,
        pool_stride: int,
        emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.temporal = nn.Conv2d(
            1,
            temporal_filters,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, temporal_padding),
            bias=False,
        )
        self.spatial = nn.Conv2d(
            temporal_filters,
            spatial_filters,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(spatial_filters)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_filters, emb_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def safe_log(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = self.safe_log(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        return self.projection(x)


class TeacherLongitudinalShallowConvNet3T(nn.Module):
    """Three-timepoint teacher using shared ShallowConvNet branches."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        spatial_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        pool_size: int,
        pool_stride: int,
        emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.branch = ShallowConvNetBranch(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            spatial_filters=spatial_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
            emb_dim=emb_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 6, 96),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = self.branch(x_t0)
        z1 = self.branch(x_t1)
        z2 = self.branch(x_t2)
        joint = torch.cat([z0, z1, z2, z1 - z0, z2 - z1, z2 - z0], dim=1)
        logits = self.classifier(joint).squeeze(1)
        return logits, joint


class StudentLongitudinalShallowConvNet2T(nn.Module):
    """Two-timepoint student used for deployment with T0/T1 only."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        spatial_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        pool_size: int,
        pool_stride: int,
        emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.branch = ShallowConvNetBranch(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            spatial_filters=spatial_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
            emb_dim=emb_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 3, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z0 = self.branch(x_t0)
        z1 = self.branch(x_t1)
        joint = torch.cat([z0, z1, z1 - z0], dim=1)
        logits = self.classifier(joint).squeeze(1)
        return logits, joint


class TeacherStudentShallowConvNet(nn.Module):
    """Teacher, student, and projection head for ShallowConvNet distillation."""

    def __init__(
        self,
        n_channels: int,
        temporal_filters: int,
        spatial_filters: int,
        temporal_kernel_size: int,
        temporal_padding: int,
        pool_size: int,
        pool_stride: int,
        emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.teacher = TeacherLongitudinalShallowConvNet3T(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            spatial_filters=spatial_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
            emb_dim=emb_dim,
            dropout=dropout,
        )
        self.student = StudentLongitudinalShallowConvNet2T(
            n_channels=n_channels,
            temporal_filters=temporal_filters,
            spatial_filters=spatial_filters,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            pool_size=pool_size,
            pool_stride=pool_stride,
            emb_dim=emb_dim,
            dropout=dropout,
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
        teacher_proj = self.teacher_projector(teacher_joint)
        return teacher_logits, teacher_proj, student_logits, student_joint

    def forward_student(self, x_t0: torch.Tensor, x_t1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student(x_t0, x_t1)


def train_one_fold(
    train_subjects,
    val_subjects: set[str],
    test_subject,
    temporal_filters: int,
    spatial_filters: int,
    temporal_kernel_size: int,
    temporal_padding: int,
    pool_size: int,
    pool_stride: int,
    emb_dim: int,
    dropout: float,
    n_aug_per_subject: int,
    train_trials_per_avg: int,
    n_test_votes: int,
    test_trials_per_avg: int,
    train_noise_std: float,
    noisy_channels_per_sample: int,
    seed: int,
) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Train one LOSO fold and return student prediction outputs."""
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

    model = TeacherStudentShallowConvNet(
        n_channels=train_x0.shape[1],
        temporal_filters=temporal_filters,
        spatial_filters=spatial_filters,
        temporal_kernel_size=temporal_kernel_size,
        temporal_padding=temporal_padding,
        pool_size=pool_size,
        pool_stride=pool_stride,
        emb_dim=emb_dim,
        dropout=dropout,
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

        teacher_logits, teacher_proj, student_logits, student_joint = model.forward_teacher_student(
            train_x0, train_x1, train_x2
        )
        teacher_cls = criterion(teacher_logits, train_y)
        student_cls = criterion(student_logits, train_y)
        kd_loss = soft_binary_kl(student_logits, teacher_logits.detach(), DISTILL_TEMPERATURE)
        feat_loss = F.mse_loss(student_joint, teacher_proj.detach())
        total_loss = teacher_cls + ALPHA * student_cls + BETA * kd_loss + GAMMA * feat_loss

        total_loss.backward()
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

    return prob, vote_probs, pd.DataFrame(history_rows), vote_debug_df


def build_output_dir(args: argparse.Namespace) -> Path:
    return BASE_OUTPUT_DIR / (
        f"tf_{args.temporal_filters}_sf{args.spatial_filters}_"
        f"k{args.temporal_kernel_size}_p{args.temporal_padding}_"
        f"pool{args.pool_size}s{args.pool_stride}_e{args.emb_dim}_"
        f"a{args.n_aug_per_subject}_tt{args.train_trials_per_avg}_"
        f"v{args.n_test_votes}_tv{args.test_trials_per_avg}_"
        f"n{str(args.train_noise_std).replace('.', 'p')}"
        f"{'_ea' if args.use_ea else ''}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-filters", type=int, default=40)
    parser.add_argument("--spatial-filters", type=int, default=40)
    parser.add_argument("--temporal-kernel-size", type=int, default=25)
    parser.add_argument("--temporal-padding", type=int, default=12)
    parser.add_argument("--pool-size", type=int, default=25)
    parser.add_argument("--pool-stride", type=int, default=5)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n-aug-per-subject", type=int, default=16)
    parser.add_argument("--train-trials-per-avg", type=int, default=16)
    parser.add_argument("--n-test-votes", type=int, default=16)
    parser.add_argument("--test-trials-per-avg", type=int, default=16)
    parser.add_argument("--train-noise-std", type=float, default=0.2)
    parser.add_argument("--noisy-channels-per-sample", type=int, default=15)
    parser.add_argument("--decision-threshold", type=float, default=0.55)
    parser.add_argument("--use-ea", action="store_true", help="Apply subject-timepoint Euclidean Alignment.")
    parser.add_argument("--ea-eps", type=float, default=1e-6)
    args = parser.parse_args()

    set_seed(SEED)
    output_dir = build_output_dir(args)
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
        f"Running teacher-student augmented LOSO ShallowConvNet | subjects={len(subjects)} | "
        f"temporal_filters={args.temporal_filters} | spatial_filters={args.spatial_filters} | "
        f"kernel={args.temporal_kernel_size} | padding={args.temporal_padding} | "
        f"pool={args.pool_size}/{args.pool_stride} | emb_dim={args.emb_dim} | "
        f"dropout={args.dropout} | aug={args.n_aug_per_subject} | "
        f"train_trials_per_avg={args.train_trials_per_avg} | test_votes={args.n_test_votes} | "
        f"test_trials_per_avg={args.test_trials_per_avg} | train_noise_std={args.train_noise_std} | "
        f"noisy_channels={args.noisy_channels_per_sample} | use_ea={args.use_ea} | "
        f"decision_threshold={args.decision_threshold}"
    )
    if skipped_subjects:
        print(f"Skipped {len(skipped_subjects)} subjects without complete T0/T1/T2 data.")

    progress = tqdm(enumerate(subjects, start=1), total=len(subjects), desc="LOSO folds", unit="fold")
    for i, test_subject in progress:
        train_subjects = [s for s in subjects if s.subject != test_subject.subject]
        val_subjects = select_val_subjects(train_subjects)
        print(
            f"[Fold {i:02d}/{len(subjects):02d}] test_subject={test_subject.subject} "
            f"true_label={'effective' if test_subject.label == 1 else 'ineffective'} | "
            f"train_subjects={len(train_subjects)} | val_subjects={len(val_subjects)}"
        )

        prob, vote_probs, history_df, vote_debug_df = train_one_fold(
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subject=test_subject,
            temporal_filters=args.temporal_filters,
            spatial_filters=args.spatial_filters,
            temporal_kernel_size=args.temporal_kernel_size,
            temporal_padding=args.temporal_padding,
            pool_size=args.pool_size,
            pool_stride=args.pool_stride,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            n_aug_per_subject=args.n_aug_per_subject,
            train_trials_per_avg=args.train_trials_per_avg,
            n_test_votes=args.n_test_votes,
            test_trials_per_avg=args.test_trials_per_avg,
            train_noise_std=args.train_noise_std,
            noisy_channels_per_sample=args.noisy_channels_per_sample,
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
    y_pred_arr = (prob_arr >= args.decision_threshold).astype(int)
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
        "teacher_input_timepoints": TEACHER_INPUT_TIMEPOINTS,
        "student_input_timepoints": STUDENT_INPUT_TIMEPOINTS,
        "decision_threshold": args.decision_threshold,
        "input_type": "teacher uses T0/T1/T2 augmented ERP samples during training; student uses T0/T1 during training/testing",
        "architecture": "teacher-student shared-branch ShallowConvNet",
        "temporal_filters": args.temporal_filters,
        "spatial_filters": args.spatial_filters,
        "temporal_kernel_size": args.temporal_kernel_size,
        "temporal_padding": args.temporal_padding,
        "pool_size": args.pool_size,
        "pool_stride": args.pool_stride,
        "emb_dim": args.emb_dim,
        "dropout": args.dropout,
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
                "teacher_input_timepoints": TEACHER_INPUT_TIMEPOINTS,
                "student_input_timepoints": STUDENT_INPUT_TIMEPOINTS,
                "decision_threshold": args.decision_threshold,
                "architecture": "teacher-student shared-branch ShallowConvNet",
                "temporal_filters": args.temporal_filters,
                "spatial_filters": args.spatial_filters,
                "temporal_kernel_size": args.temporal_kernel_size,
                "temporal_padding": args.temporal_padding,
                "pool_size": args.pool_size,
                "pool_stride": args.pool_stride,
                "emb_dim": args.emb_dim,
                "dropout": args.dropout,
                "train_noise_std": args.train_noise_std,
                "noisy_channels_per_sample": args.noisy_channels_per_sample,
                "use_ea": args.use_ea,
                "ea_mode": "subject_timepoint" if args.use_ea else "none",
                "ea_eps": args.ea_eps,
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
