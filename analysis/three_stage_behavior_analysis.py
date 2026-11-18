"""Generate ANT behavioral summaries, figures, and T0-to-T2 group comparisons.

The script keeps the original experimental-vs-control three-stage analysis and
adds a T0-to-T2 comparison among:

1. experimental_effective
2. experimental_ineffective
3. control

Positive improvement means better performance:
- accuracy: T2 - T0
- RT, RT variability, impulsive rate, omission rate: T0 - T2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL_CSV = ROOT / "data/behavior/probiotic_behavior.csv"
CONTROL_CSV = ROOT / "data/behavior/control_behavior.csv"
RESPONSE_CSV = ROOT / "data/behavior/response_labels.csv"
OUT_DIR = ROOT / "outputs/behavior"
TIMEPOINTS = ["T0", "T1", "T2"]
METRICS = {
    "mean_rt_ms": {"label": "Mean RT (ms)", "direction": "lower_better"},
    "accuracy_pct": {"label": "Accuracy (%)", "direction": "higher_better"},
    "rt_variability_cv_pct": {"label": "RT variability CV (%)", "direction": "lower_better"},
    "impulsive_rate_pct": {"label": "Impulsive rate (%)", "direction": "lower_better"},
    "omission_rate_pct": {"label": "Omission rate (%)", "direction": "lower_better"},
}
GROUP_ORDER = ["experimental_effective", "experimental_ineffective", "control"]
GROUP_LABELS = {
    "experimental_effective": "Effective",
    "experimental_ineffective": "Ineffective",
    "control": "Control",
}
GROUP_COLORS = {
    "experimental_effective": "#1f77b4",
    "experimental_ineffective": "#ff7f0e",
    "control": "#2ca02c",
}


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    temp = np.empty_like(adjusted)
    temp[order] = adjusted
    out[valid] = temp
    return out


def load_group(path: Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "group", group)
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for tp in TIMEPOINTS:
            out = {"group": row["group"], "subject": row["subject"], "timepoint": tp}
            for metric in METRICS:
                out[metric] = row.get(f"{tp}_{metric}", np.nan)
            rows.append(out)
    return pd.DataFrame(rows)


def paired_stats(values_a: pd.Series, values_b: pd.Series) -> dict[str, float]:
    paired = pd.concat([values_a, values_b], axis=1).dropna()
    if len(paired) < 2:
        return {"n": len(paired), "t_stat": np.nan, "t_p": np.nan, "wilcoxon_p": np.nan, "dz": np.nan}
    diff = paired.iloc[:, 1] - paired.iloc[:, 0]
    t_stat, t_p = stats.ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1], nan_policy="omit")
    try:
        _, w_p = stats.wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1])
    except ValueError:
        w_p = np.nan
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    return {"n": len(paired), "t_stat": t_stat, "t_p": t_p, "wilcoxon_p": w_p, "dz": dz}


def group_time_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ["experimental", "control"]:
        sub = long_df[long_df["group"] == group]
        for tp in TIMEPOINTS:
            tp_df = sub[sub["timepoint"] == tp]
            for metric, meta in METRICS.items():
                values = tp_df[metric].dropna()
                rows.append(
                    {
                        "group": group,
                        "timepoint": tp,
                        "metric": metric,
                        "metric_label": meta["label"],
                        "direction": meta["direction"],
                        "n": int(values.shape[0]),
                        "mean": float(values.mean()) if len(values) else np.nan,
                        "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                        "sem": float(values.sem()) if len(values) > 1 else np.nan,
                        "median": float(values.median()) if len(values) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def paired_change_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("T0", "T1"), ("T1", "T2"), ("T0", "T2")]
    rows = []
    for group in ["experimental", "control"]:
        sub = wide_df[wide_df["group"] == group].set_index("subject")
        for metric, meta in METRICS.items():
            for start, end in comparisons:
                a = sub[f"{start}_{metric}"]
                b = sub[f"{end}_{metric}"]
                raw_change = b - a
                improvement = raw_change if meta["direction"] == "higher_better" else -raw_change
                st = paired_stats(a, b)
                rows.append(
                    {
                        "group": group,
                        "comparison": f"{start}_to_{end}",
                        "metric": metric,
                        "metric_label": meta["label"],
                        "direction": meta["direction"],
                        "n": st["n"],
                        f"{start}_mean": float(a.mean()),
                        f"{end}_mean": float(b.mean()),
                        "raw_change_mean_end_minus_start": float(raw_change.mean()),
                        "improvement_mean": float(improvement.mean()),
                        "improvement_std": float(improvement.std(ddof=1)),
                        "paired_t_stat": st["t_stat"],
                        "paired_t_p": st["t_p"],
                        "wilcoxon_p": st["wilcoxon_p"],
                        "paired_effect_dz": st["dz"],
                    }
                )
    return pd.DataFrame(rows)


def between_group_change_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("T0", "T1"), ("T1", "T2"), ("T0", "T2")]
    rows = []
    for metric, meta in METRICS.items():
        for start, end in comparisons:
            group_values = {}
            for group in ["experimental", "control"]:
                sub = wide_df[wide_df["group"] == group]
                raw = sub[f"{end}_{metric}"] - sub[f"{start}_{metric}"]
                imp = raw if meta["direction"] == "higher_better" else -raw
                group_values[group] = imp.dropna()
            exp = group_values["experimental"]
            ctrl = group_values["control"]
            t_stat, t_p = stats.ttest_ind(exp, ctrl, equal_var=False, nan_policy="omit")
            u_stat, u_p = stats.mannwhitneyu(exp, ctrl, alternative="two-sided")
            pooled = np.sqrt(
                ((len(exp) - 1) * exp.var(ddof=1) + (len(ctrl) - 1) * ctrl.var(ddof=1))
                / max(len(exp) + len(ctrl) - 2, 1)
            )
            cohen_d = (exp.mean() - ctrl.mean()) / pooled if pooled > 0 else np.nan
            correction = 1 - (3 / (4 * (len(exp) + len(ctrl)) - 9))
            rows.append(
                {
                    "comparison": f"{start}_to_{end}",
                    "metric": metric,
                    "metric_label": meta["label"],
                    "direction": meta["direction"],
                    "experimental_n": int(len(exp)),
                    "control_n": int(len(ctrl)),
                    "experimental_improvement_mean": float(exp.mean()),
                    "control_improvement_mean": float(ctrl.mean()),
                    "improvement_mean_difference": float(exp.mean() - ctrl.mean()),
                    "welch_t_stat": t_stat,
                    "welch_t_p": t_p,
                    "mannwhitney_u_stat": u_stat,
                    "mannwhitney_u_p": u_p,
                    "hedges_g": cohen_d * correction if not np.isnan(cohen_d) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_bh_p"] = fdr_bh(out["welch_t_p"].to_numpy())
    return out


def build_response_control_t0t2_change(wide_df: pd.DataFrame) -> pd.DataFrame:
    response = pd.read_csv(RESPONSE_CSV)[["subject", "final_response_label_core2"]]
    exp = wide_df[wide_df["group"] == "experimental"].merge(response, on="subject", how="left")
    exp["comparison_group"] = exp["final_response_label_core2"].map(
        {"effective": "experimental_effective", "ineffective": "experimental_ineffective"}
    )
    ctrl = wide_df[wide_df["group"] == "control"].copy()
    ctrl["comparison_group"] = "control"
    ctrl["final_response_label_core2"] = "control"
    combined = pd.concat([exp, ctrl], ignore_index=True)

    rows = []
    for _, row in combined.iterrows():
        if pd.isna(row.get("comparison_group")):
            continue
        for metric, meta in METRICS.items():
            t0 = row.get(f"T0_{metric}", np.nan)
            t2 = row.get(f"T2_{metric}", np.nan)
            raw_change = t2 - t0
            improvement = raw_change if meta["direction"] == "higher_better" else -raw_change
            rows.append(
                {
                    "subject": row["subject"],
                    "comparison_group": row["comparison_group"],
                    "group_label": GROUP_LABELS[row["comparison_group"]],
                    "original_group": row["group"],
                    "response_label": row["final_response_label_core2"],
                    "metric": metric,
                    "metric_label": meta["label"],
                    "direction": meta["direction"],
                    "T0": t0,
                    "T2": t2,
                    "raw_change_T2_minus_T0": raw_change,
                    "improvement_T0_to_T2": improvement,
                }
            )
    return pd.DataFrame(rows)


def response_control_t0t2_group_summary(change_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, meta in METRICS.items():
        for group in GROUP_ORDER:
            values = change_df[
                (change_df["metric"] == metric) & (change_df["comparison_group"] == group)
            ]["improvement_T0_to_T2"].dropna()
            rows.append(
                {
                    "comparison_group": group,
                    "group_label": GROUP_LABELS[group],
                    "metric": metric,
                    "metric_label": meta["label"],
                    "direction": meta["direction"],
                    "n": int(values.shape[0]),
                    "improvement_mean": float(values.mean()) if len(values) else np.nan,
                    "improvement_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "improvement_sem": float(values.sem()) if len(values) > 1 else np.nan,
                    "improvement_median": float(values.median()) if len(values) else np.nan,
                    "improvement_iqr": float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def response_control_t0t2_between_summary(change_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_order = [
        ("experimental_effective", "experimental_ineffective"),
        ("experimental_effective", "control"),
        ("experimental_ineffective", "control"),
    ]
    omnibus_rows = []
    pair_rows = []
    for metric, meta in METRICS.items():
        metric_df = change_df[change_df["metric"] == metric]
        arrays = [
            metric_df[metric_df["comparison_group"] == group]["improvement_T0_to_T2"].dropna()
            for group in GROUP_ORDER
        ]
        if all(len(arr) > 0 for arr in arrays):
            h_stat, h_p = stats.kruskal(*arrays)
        else:
            h_stat, h_p = np.nan, np.nan
        omnibus_rows.append(
            {
                "metric": metric,
                "metric_label": meta["label"],
                "direction": meta["direction"],
                "test": "kruskal_wallis",
                "h_stat": h_stat,
                "p_value": h_p,
            }
        )
        for group_a, group_b in pair_order:
            a = metric_df[metric_df["comparison_group"] == group_a]["improvement_T0_to_T2"].dropna()
            b = metric_df[metric_df["comparison_group"] == group_b]["improvement_T0_to_T2"].dropna()
            if len(a) > 1 and len(b) > 1:
                t_stat, t_p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                pooled = np.sqrt(
                    ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / max(len(a) + len(b) - 2, 1)
                )
                cohen_d = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
                correction = 1 - (3 / (4 * (len(a) + len(b)) - 9))
                hedges_g = cohen_d * correction if not np.isnan(cohen_d) else np.nan
            else:
                t_stat = t_p = u_stat = u_p = hedges_g = np.nan
            pair_rows.append(
                {
                    "metric": metric,
                    "metric_label": meta["label"],
                    "direction": meta["direction"],
                    "group_a": group_a,
                    "group_b": group_b,
                    "group_a_label": GROUP_LABELS[group_a],
                    "group_b_label": GROUP_LABELS[group_b],
                    "group_a_n": int(len(a)),
                    "group_b_n": int(len(b)),
                    "group_a_improvement_mean": float(a.mean()) if len(a) else np.nan,
                    "group_b_improvement_mean": float(b.mean()) if len(b) else np.nan,
                    "mean_difference_a_minus_b": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
                    "welch_t_stat": t_stat,
                    "welch_t_p": t_p,
                    "mannwhitney_u_stat": u_stat,
                    "mannwhitney_u_p": u_p,
                    "hedges_g": hedges_g,
                }
            )
    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pair_rows)
    omnibus["fdr_bh_p"] = fdr_bh(omnibus["p_value"].to_numpy())
    pairwise["fdr_bh_p"] = fdr_bh(pairwise["welch_t_p"].to_numpy())
    return omnibus, pairwise


def plot_metric_trajectories(summary_df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(TIMEPOINTS))
    colors = {"experimental": "#1f77b4", "control": "#d62728"}
    for metric, meta in METRICS.items():
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for group in ["experimental", "control"]:
            sub = summary_df[
                (summary_df["group"] == group) & (summary_df["metric"] == metric)
            ].set_index("timepoint").loc[TIMEPOINTS]
            mean = sub["mean"].to_numpy()
            sem = sub["sem"].to_numpy()
            ax.plot(x, mean, marker="o", linewidth=2.2, color=colors[group], label=group)
            ax.fill_between(x, mean - sem, mean + sem, color=colors[group], alpha=0.18)
        ax.set_xticks(x)
        ax.set_xticklabels(TIMEPOINTS)
        ax.set_ylabel(meta["label"])
        ax.set_xlabel("Timepoint")
        ax.set_title(f"Three-stage trajectory: {meta['label']}")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(output_dir / f"three_stage_{metric}_trajectory.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_all_metrics(summary_df: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(TIMEPOINTS))
    colors = {"experimental": "#1f77b4", "control": "#d62728"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (metric, meta) in zip(axes, METRICS.items()):
        for group in ["experimental", "control"]:
            sub = summary_df[
                (summary_df["group"] == group) & (summary_df["metric"] == metric)
            ].set_index("timepoint").loc[TIMEPOINTS]
            mean = sub["mean"].to_numpy()
            sem = sub["sem"].to_numpy()
            ax.plot(x, mean, marker="o", linewidth=2.0, color=colors[group], label=group)
            ax.fill_between(x, mean - sem, mean + sem, color=colors[group], alpha=0.16)
        ax.set_xticks(x)
        ax.set_xticklabels(TIMEPOINTS)
        ax.set_title(meta["label"])
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    fig.suptitle("ANT behavioral metrics across T0, T1, and T2", fontsize=15)
    fig.savefig(output_dir / "ant_three_stage_behavior_trajectories.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "ant_three_stage_behavior_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_response_control_t0t2_boxplots(change_df: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (metric, meta) in zip(axes, METRICS.items()):
        data = [
            change_df[
                (change_df["metric"] == metric) & (change_df["comparison_group"] == group)
            ]["improvement_T0_to_T2"].dropna().to_numpy()
            for group in GROUP_ORDER
        ]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            labels=[GROUP_LABELS[group] for group in GROUP_ORDER],
        )
        for patch, group in zip(bp["boxes"], GROUP_ORDER):
            patch.set_facecolor(GROUP_COLORS[group])
            patch.set_alpha(0.35)
            patch.set_edgecolor(GROUP_COLORS[group])
        for idx, (values, group) in enumerate(zip(data, GROUP_ORDER), start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(np.full(len(values), idx) + jitter, values, s=28, color=GROUP_COLORS[group], alpha=0.75)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(meta["label"])
        ax.set_ylabel("Improvement (T0 to T2)")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS[group], label=GROUP_LABELS[group], markersize=8)
        for group in GROUP_ORDER
    ]
    axes[-1].legend(handles=handles, loc="center", frameon=False, fontsize=12)
    fig.suptitle("T0-to-T2 ANT improvement: effective vs ineffective vs control", fontsize=15)
    fig.savefig(output_dir / "ant_t0_t2_response_control_improvement_boxplots.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "ant_t0_t2_response_control_improvement_boxplots.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, three_group_pairwise: pd.DataFrame, three_group_omnibus: pd.DataFrame) -> None:
    lines = [
        "ANT Behavioral Results Across Three Timepoints",
        "",
        "The original experimental-vs-control outputs summarize T0, T1, and T2 trajectories.",
        "This updated analysis additionally compares T0-to-T2 behavioral improvement among experimental effective, experimental ineffective, and control groups.",
        "",
        "Improvement definition:",
        "- Accuracy: T2 minus T0.",
        "- RT, RT variability, impulsive rate, and omission rate: T0 minus T2.",
        "",
        "Three-group omnibus tests (Kruskal-Wallis):",
    ]
    for row in three_group_omnibus.itertuples(index=False):
        lines.append(f"- {row.metric_label}: H = {row.h_stat:.3f}, p = {row.p_value:.4f}, FDR p = {row.fdr_bh_p:.4f}.")
    lines.append("")
    lines.append("Pairwise Welch-test results are saved in ant_t0_t2_response_control_pairwise_summary.csv.")
    (output_dir / "ant_three_stage_behavior_report.doc").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp = load_group(EXPERIMENTAL_CSV, "experimental")
    ctrl = load_group(CONTROL_CSV, "control")
    wide = pd.concat([exp, ctrl], ignore_index=True)

    long_df = wide_to_long(wide)
    summary = group_time_summary(long_df)
    paired = paired_change_summary(wide)
    between = between_group_change_summary(wide)

    response_control_change = build_response_control_t0t2_change(wide)
    response_control_summary = response_control_t0t2_group_summary(response_control_change)
    response_control_omnibus, response_control_pairwise = response_control_t0t2_between_summary(response_control_change)

    long_df.to_csv(OUT_DIR / "ant_three_stage_behavior_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "ant_three_stage_group_time_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT_DIR / "ant_three_stage_paired_change_summary.csv", index=False, encoding="utf-8-sig")
    between.to_csv(OUT_DIR / "ant_three_stage_between_group_change_summary.csv", index=False, encoding="utf-8-sig")

    response_control_change.to_csv(
        OUT_DIR / "ant_t0_t2_response_control_change_by_subject.csv",
        index=False,
        encoding="utf-8-sig",
    )
    response_control_summary.to_csv(
        OUT_DIR / "ant_t0_t2_response_control_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    response_control_omnibus.to_csv(
        OUT_DIR / "ant_t0_t2_response_control_omnibus_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    response_control_pairwise.to_csv(
        OUT_DIR / "ant_t0_t2_response_control_pairwise_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_metric_trajectories(summary, OUT_DIR)
    plot_all_metrics(summary, OUT_DIR)
    plot_response_control_t0t2_boxplots(response_control_change, OUT_DIR)
    write_report(OUT_DIR, response_control_pairwise, response_control_omnibus)
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
