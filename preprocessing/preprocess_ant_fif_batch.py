from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np


TARGET_EVENT_VALUES = set(range(200, 212))
START_EVENT_CODES = {"0001", "1"}
REMOVE_CHANNELS = {"M1", "M2"}
EXPECTED_CHANNELS = [
    "Fp1",
    "Fpz",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "M1",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "M2",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "Oz",
    "O2",
]


@dataclass
class PreprocessSummary:
    input_file: str
    output_file: str
    sfreq_before: float
    sfreq_after: float
    duration_before_sec: float
    duration_after_sec: float
    n_channels_before: int
    n_channels_after: int
    first_start_event_onset_sec: float | None
    removed_channels: list[str]
    bad_channels_detected: list[str]
    bad_channels_interpolated: list[str]
    ica_components_removed: list[int]
    ica_label_status: str
    ica_label_error: str | None
    ica_component_details: list[dict[str, float | int | str]]
    event_code_counts: dict[str, int]
    target_event_count_before_limit: int
    target_event_count_used: int
    rejected_epoch_indices: list[int]
    remaining_epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch preprocess ANT FIF files and save preprocessed outputs."
    )
    parser.add_argument(
        "--input-root",
        default=r"restricted_data/raw",
        help=r"Root directory containing FIF files. Default: restricted_data/raw",
    )
    parser.add_argument(
        "--output-root",
        default=r"restricted_data/ADHDtime_preprocessed",
        help=r"Directory to save preprocessed files. Default: restricted_data/ADHDtime_preprocessed",
    )
    parser.add_argument(
        "--pattern",
        default="*_ANT.fif",
        help="Filename glob to include. Default: *_ANT.fif",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--resample",
        type=float,
        default=250.0,
        help="Target sampling rate. Default: 250 Hz.",
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=0.5,
        help="High-pass cutoff. Default: 0.5 Hz.",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=45.0,
        help="Low-pass cutoff. Default: 45 Hz.",
    )
    parser.add_argument(
        "--epoch-tmin",
        type=float,
        default=-0.2,
        help="Epoch start in seconds. Default: -0.2.",
    )
    parser.add_argument(
        "--epoch-tmax",
        type=float,
        default=0.996,
        help="Epoch end in seconds. Default: 0.996 (300 samples at 250 Hz with tmin=-0.2).",
    )
    parser.add_argument(
        "--baseline-start",
        type=float,
        default=-0.2,
        help="Baseline start in seconds. Default: -0.2.",
    )
    parser.add_argument(
        "--baseline-end",
        type=float,
        default=0.0,
        help="Baseline end in seconds. Default: 0.0.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=288,
        help="Maximum number of target epochs to keep. Default: 288.",
    )
    parser.add_argument(
        "--amplitude-threshold-uv",
        type=float,
        default=150.0,
        help="Epoch rejection threshold in microvolts. Default: 150.",
    )
    parser.add_argument(
        "--ica-highpass",
        type=float,
        default=1.0,
        help="High-pass cutoff used to fit ICA. Default: 1 Hz.",
    )
    parser.add_argument(
        "--ica-lowpass",
        type=float,
        default=100.0,
        help="Low-pass cutoff used to fit ICA. Default: 100 Hz.",
    )
    parser.add_argument(
        "--ica-max-components",
        type=int,
        default=29,
        help="Maximum ICA PCA components. Default: 29.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for number of files to process. 0 means all.",
    )
    return parser.parse_args()


def normalize_event_code(desc: str) -> str:
    desc = str(desc).strip()
    if desc.isdigit():
        return f"{int(desc):04d}"
    try:
        value = float(desc)
        if value.is_integer():
            return f"{int(value):04d}"
    except Exception:
        pass
    return desc


def event_code_to_int(desc: str) -> int | None:
    desc = normalize_event_code(desc)
    try:
        value = float(desc)
        if value.is_integer():
            return int(value)
    except Exception:
        return None
    return None


def crop_from_first_start_event(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, float | None]:
    descriptions = [normalize_event_code(desc) for desc in raw.annotations.description]
    start_idx = next((idx for idx, desc in enumerate(descriptions) if desc in START_EVENT_CODES), None)
    if start_idx is None:
        return raw, None

    start_onset = float(raw.annotations.onset[start_idx])
    if start_onset <= 0:
        return raw, start_onset

    cropped = raw.copy().crop(tmin=start_onset, tmax=None)
    return cropped, start_onset


def ensure_channel_names(raw: mne.io.BaseRaw) -> None:
    if all(name and str(name).strip() for name in raw.ch_names):
        return

    rename_map = {}
    for idx, old_name in enumerate(raw.ch_names):
        if not old_name or not str(old_name).strip():
            if idx < len(EXPECTED_CHANNELS):
                rename_map[old_name] = EXPECTED_CHANNELS[idx]
    if rename_map:
        mne.rename_channels(raw.info, rename_map, allow_duplicates=False)


def apply_montage_if_possible(raw: mne.io.BaseRaw) -> None:
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass


def detect_bad_channels(raw: mne.io.BaseRaw) -> list[str]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return []

    data = raw.get_data(picks=picks)
    channel_std = np.std(data, axis=1)
    channel_ptp = np.ptp(data, axis=1)

    bad_mask = np.zeros(len(picks), dtype=bool)
    if np.all(channel_std > 0):
        log_std = np.log(channel_std + np.finfo(float).eps)
        z_std = (log_std - np.mean(log_std)) / (np.std(log_std) + np.finfo(float).eps)
        bad_mask |= np.abs(z_std) > 3.0
    if np.all(channel_ptp > 0):
        log_ptp = np.log(channel_ptp + np.finfo(float).eps)
        z_ptp = (log_ptp - np.mean(log_ptp)) / (np.std(log_ptp) + np.finfo(float).eps)
        bad_mask |= np.abs(z_ptp) > 3.0

    return [raw.ch_names[picks[idx]] for idx in np.where(bad_mask)[0]]


def limit_target_events(events: np.ndarray, max_epochs: int) -> np.ndarray:
    if max_epochs > 0 and len(events) > max_epochs:
        return events[:max_epochs]
    return events


def create_epochs(
    raw: mne.io.BaseRaw,
    target_event_values: set[int],
    tmin: float,
    tmax: float,
    baseline: tuple[float, float],
    max_epochs: int,
) -> tuple[mne.Epochs, int, int]:
    if len(raw.annotations) == 0:
        raise RuntimeError("No annotations found in file.")

    target_desc = sorted(
        {
            str(desc).strip()
            for desc in raw.annotations.description
            if event_code_to_int(str(desc).strip()) in target_event_values
        }
    )
    if not target_desc:
        raise RuntimeError("No target events 0200-0211 found.")
    event_desc = {desc: event_code_to_int(desc) for desc in target_desc}

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=event_desc,
        verbose="ERROR",
    )
    if len(events) == 0:
        raise RuntimeError("Target events were mapped but no events were extracted.")

    used_events = limit_target_events(events, max_epochs=max_epochs)
    epochs = mne.Epochs(
        raw,
        used_events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    return epochs, len(events), len(used_events)


def reject_epochs_by_amplitude(epochs: mne.Epochs, threshold_uv: float) -> tuple[mne.Epochs, list[int]]:
    if len(epochs) == 0:
        return epochs, []

    threshold_v = threshold_uv * 1e-6
    data = epochs.get_data(copy=False)
    peak = np.max(np.abs(data), axis=(1, 2))
    rejected = np.where(peak > threshold_v)[0].tolist()

    if rejected and len(rejected) < len(epochs):
        kept = np.setdiff1d(np.arange(len(epochs)), rejected)
        epochs = epochs[kept]
    elif len(rejected) == len(epochs):
        rejected = []
    return epochs, rejected


def _fit_and_apply_ica_with_details(
    raw_to_clean: mne.io.BaseRaw,
    ica_source: mne.io.BaseRaw,
    highpass: float,
    lowpass: float,
    max_components: int,
) -> tuple[mne.io.BaseRaw, list[int], str, str | None, list[dict[str, float | int | str]]]:
    eeg_picks = mne.pick_types(ica_source.info, eeg=True, exclude="bads")
    if len(eeg_picks) < 5:
        return raw_to_clean, [], "skipped", "Not enough EEG channels for ICA.", []

    ica_raw = ica_source.copy().filter(l_freq=highpass, h_freq=lowpass, verbose="ERROR")
    n_components = min(max_components, len(eeg_picks) - 1)
    if n_components < 2:
        return raw_to_clean, [], "skipped", "Not enough components for ICA.", []

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=97,
        max_iter="auto",
    )
    ica.fit(ica_raw, picks=eeg_picks, verbose="ERROR")

    try:
        from mne_icalabel import label_components

        labels = label_components(ica_raw, ica, method="iclabel")
        probabilities = labels["y_pred_proba"]
        predicted_labels = labels["labels"]
        exclude = []
        details: list[dict[str, float | int | str]] = []
        for idx, (label_name, prob) in enumerate(zip(predicted_labels, probabilities)):
            label_name = str(label_name).lower()
            prob = float(prob)
            details.append(
                {
                    "component": idx,
                    "label": label_name,
                    "probability": prob,
                }
            )
            if label_name != "brain" and prob >= 0.7:
                exclude.append(idx)
        status = "ok"
        error = None
    except Exception as exc:
        exclude = []
        details = []
        status = "failed"
        error = str(exc)

    if exclude:
        raw_to_clean = ica.apply(raw_to_clean.copy(), exclude=exclude, verbose="ERROR")
    return raw_to_clean, exclude, status, error, details


def preprocess_file(file_path: Path, output_root: Path, args: argparse.Namespace) -> PreprocessSummary:
    raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose="ERROR")
    summary = PreprocessSummary(
        input_file=str(file_path),
        output_file="",
        sfreq_before=float(raw.info["sfreq"]),
        sfreq_after=0.0,
        duration_before_sec=float(raw.n_times / raw.info["sfreq"]),
        duration_after_sec=0.0,
        n_channels_before=int(raw.info["nchan"]),
        n_channels_after=0,
        first_start_event_onset_sec=None,
        removed_channels=[],
        bad_channels_detected=[],
        bad_channels_interpolated=[],
        ica_components_removed=[],
        ica_label_status="not_run",
        ica_label_error=None,
        ica_component_details=[],
        event_code_counts={},
        target_event_count_before_limit=0,
        target_event_count_used=0,
        rejected_epoch_indices=[],
        remaining_epochs=0,
    )

    raw, first_start_onset = crop_from_first_start_event(raw)
    summary.first_start_event_onset_sec = first_start_onset

    if raw.info["sfreq"] > args.resample:
        raw.resample(args.resample, npad="auto", verbose="ERROR")

    ensure_channel_names(raw)
    apply_montage_if_possible(raw)

    to_drop = [ch for ch in raw.ch_names if ch.upper() in REMOVE_CHANNELS]
    if to_drop:
        raw.drop_channels(to_drop)
    summary.removed_channels = to_drop

    summary.bad_channels_detected = detect_bad_channels(raw)
    if summary.bad_channels_detected:
        raw.info["bads"] = summary.bad_channels_detected.copy()
        raw.interpolate_bads(reset_bads=False, verbose="ERROR")
        summary.bad_channels_interpolated = summary.bad_channels_detected.copy()

    ica_source = raw.copy()
    raw.filter(l_freq=args.highpass, h_freq=args.lowpass, verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
    ica_source.set_eeg_reference(ref_channels="average", verbose="ERROR")

    raw, excluded_components, ica_status, ica_error, ica_details = _fit_and_apply_ica_with_details(
        raw_to_clean=raw,
        ica_source=ica_source,
        highpass=args.ica_highpass,
        lowpass=args.ica_lowpass,
        max_components=args.ica_max_components,
    )
    summary.ica_components_removed = excluded_components
    summary.ica_label_status = ica_status
    summary.ica_label_error = ica_error
    summary.ica_component_details = ica_details

    descriptions = [normalize_event_code(desc) for desc in raw.annotations.description]
    summary.event_code_counts = dict(sorted(Counter(descriptions).items()))

    epochs, total_target_events, used_target_events = create_epochs(
        raw,
        target_event_values=TARGET_EVENT_VALUES,
        tmin=args.epoch_tmin,
        tmax=args.epoch_tmax,
        baseline=(args.baseline_start, args.baseline_end),
        max_epochs=args.max_epochs,
    )
    summary.target_event_count_before_limit = total_target_events
    summary.target_event_count_used = used_target_events

    epochs, rejected_epoch_indices = reject_epochs_by_amplitude(
        epochs,
        threshold_uv=args.amplitude_threshold_uv,
    )
    summary.rejected_epoch_indices = rejected_epoch_indices
    summary.remaining_epochs = len(epochs)

    relative_path = file_path.relative_to(Path(args.input_root))
    output_relative = relative_path.with_name(f"{relative_path.stem}_epo.fif")
    output_file = output_root / output_relative
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.output_file = str(output_file)

    summary.sfreq_after = float(epochs.info["sfreq"])
    epoch_duration_sec = float(epochs.times[-1] - epochs.times[0]) if len(epochs.times) > 1 else 0.0
    summary.duration_after_sec = float(len(epochs) * epoch_duration_sec)
    summary.n_channels_after = int(epochs.info["nchan"])
    epochs.save(str(output_file), overwrite=args.overwrite)
    summary_path = output_file.with_name(f"{output_file.stem}_summary.json")
    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    fif_files = sorted(
        path
        for path in input_root.rglob(args.pattern)
        if path.is_file() and path.suffix.lower() == ".fif"
    )
    if args.max_files > 0:
        fif_files = fif_files[: args.max_files]

    print(f"Found {len(fif_files)} FIF files under {input_root}")

    success_count = 0
    failed_count = 0
    failed_files: list[dict[str, str]] = []

    for file_path in fif_files:
        try:
            summary = preprocess_file(file_path, output_root=output_root, args=args)
            success_count += 1
            print(
                f"OK    {file_path} -> {summary.output_file} "
                f"(epochs={summary.remaining_epochs}, removed={len(summary.rejected_epoch_indices)})"
            )
        except Exception as exc:
            failed_count += 1
            failed_files.append({"file": str(file_path), "error": str(exc)})
            print(f"FAIL  {file_path} -> {exc}")

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "success_count": success_count,
        "failed_count": failed_count,
        "failed_files": failed_files,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
