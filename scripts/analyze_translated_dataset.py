from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {"class_0", "class_1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisi dataset eventi clinici con lettura chunked"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/"
            "ALL_20DRG_2022_2023_CLASS_Duration_"
            "ricovero_dimissioni_LAST_17Jan2025_edited_translated.csv"
        ),
        help="Path al CSV da analizzare",
    )
    parser.add_argument(
        "--translation-cache",
        type=Path,
        default=Path("data/translation_cache.json"),
        help="JSON con mappatura activity -> traduzione",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/dataset_analysis"),
        help="Directory di output per report e grafici",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Numero di righe lette per chunk",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Numero di azioni top da salvare/visualizzare",
    )
    return parser.parse_args()


def iter_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    usecols = ["case_id", "activity", "timestamp", "translated_activity"]
    dtype = {
        "case_id": "string",
        "activity": "string",
        "timestamp": "string",
        "translated_activity": "string",
    }
    return pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        keep_default_na=False,
        chunksize=chunksize,
    )


def resolve_translation(
    activity: str,
    cache_map: Dict[str, str],
    fallback_map: Dict[str, str],
) -> str:
    if activity in cache_map:
        return cache_map[activity]
    return fallback_map.get(activity, "")


def top_actions_df(
    counts: Counter,
    total: int,
    cache_map: Dict[str, str],
    fallback_map: Dict[str, str],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for activity, n in counts.most_common(top_k):
        rows.append(
            {
                "activity": activity,
                "translation": resolve_translation(activity, cache_map, fallback_map),
                "count": n,
                "pct": (n / total * 100.0) if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File CSV non trovato: {args.input}")
    if not args.translation_cache.exists():
        raise FileNotFoundError(
            f"Translation cache non trovato: {args.translation_cache}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.translation_cache.open("r", encoding="utf-8") as f:
        translation_cache = json.load(f)

    case_row_counts: Counter = Counter()
    global_activity_counts: Counter = Counter()
    activity_translated_fallback: Dict[str, str] = {}
    last_event_by_case: Dict[str, Tuple[str, int, str]] = {}

    row_idx = 0
    for chunk in iter_chunks(args.input, args.chunksize):
        for row in chunk.itertuples(index=False):
            case_id = str(row.case_id)
            activity = str(row.activity)
            timestamp = str(row.timestamp)
            translated_activity = str(row.translated_activity)

            case_row_counts[case_id] += 1
            global_activity_counts[activity] += 1

            if translated_activity and activity not in activity_translated_fallback:
                activity_translated_fallback[activity] = translated_activity

            prev = last_event_by_case.get(case_id)
            if prev is None or (timestamp, row_idx) >= (prev[0], prev[1]):
                last_event_by_case[case_id] = (timestamp, row_idx, activity)

            row_idx += 1

    labels_by_case: Dict[str, str] = {}
    for case_id, (_, _, activity) in last_event_by_case.items():
        if activity in LABELS:
            labels_by_case[case_id] = activity

    class_event_counts = {"class_0": Counter(), "class_1": Counter()}
    class_case_presence = {
        "class_0": defaultdict(set),
        "class_1": defaultdict(set),
    }
    class_total_events = {"class_0": 0, "class_1": 0}

    for chunk in iter_chunks(args.input, args.chunksize):
        for row in chunk.itertuples(index=False):
            case_id = str(row.case_id)
            activity = str(row.activity)
            label = labels_by_case.get(case_id)

            if label not in LABELS:
                continue
            if activity in LABELS:
                continue

            class_event_counts[label][activity] += 1
            class_total_events[label] += 1
            class_case_presence[label][activity].add(case_id)

    unique_cases = len(case_row_counts)
    actions_per_case = pd.Series(case_row_counts, dtype="int64")
    mean_actions = float(actions_per_case.mean())
    median_actions = float(actions_per_case.median())

    non_label_activity_counts = Counter(
        {
            action: n
            for action, n in global_activity_counts.items()
            if action not in LABELS
        }
    )

    total_non_label_events = sum(non_label_activity_counts.values())
    df_top_overall = top_actions_df(
        non_label_activity_counts,
        total_non_label_events,
        translation_cache,
        activity_translated_fallback,
        args.top_k,
    )
    df_top_overall.to_csv(args.output_dir / "top_actions_overall.csv", index=False)

    class_cases = {
        label: [cid for cid, c_label in labels_by_case.items() if c_label == label]
        for label in LABELS
    }

    quantity_rows = []
    counts_by_class = {}
    for label in ["class_0", "class_1"]:
        case_ids = class_cases[label]
        total_counts = [case_row_counts[cid] for cid in case_ids]
        non_label_counts = [max(case_row_counts[cid] - 1, 0) for cid in case_ids]
        counts_by_class[label] = non_label_counts

        if case_ids:
            quantity_rows.append(
                {
                    "label": label,
                    "n_cases": len(case_ids),
                    "mean_actions_total": float(pd.Series(total_counts).mean()),
                    "median_actions_total": float(pd.Series(total_counts).median()),
                    "mean_actions_no_label": float(pd.Series(non_label_counts).mean()),
                    "median_actions_no_label": float(
                        pd.Series(non_label_counts).median()
                    ),
                }
            )
        else:
            quantity_rows.append(
                {
                    "label": label,
                    "n_cases": 0,
                    "mean_actions_total": 0.0,
                    "median_actions_total": 0.0,
                    "mean_actions_no_label": 0.0,
                    "median_actions_no_label": 0.0,
                }
            )

    df_quantity = pd.DataFrame(quantity_rows)
    df_quantity.to_csv(args.output_dir / "class_quantity_summary.csv", index=False)

    for label in ["class_0", "class_1"]:
        df_top_class = top_actions_df(
            class_event_counts[label],
            class_total_events[label],
            translation_cache,
            activity_translated_fallback,
            args.top_k,
        )
        df_top_class.to_csv(args.output_dir / f"top_actions_{label}.csv", index=False)

    n_cases_class_0 = len(class_cases["class_0"])
    n_cases_class_1 = len(class_cases["class_1"])

    all_actions = set(class_case_presence["class_0"].keys()) | set(
        class_case_presence["class_1"].keys()
    )
    prevalence_rows = []
    for activity in all_actions:
        p0 = (
            len(class_case_presence["class_0"][activity]) / n_cases_class_0
            if n_cases_class_0 > 0
            else 0.0
        )
        p1 = (
            len(class_case_presence["class_1"][activity]) / n_cases_class_1
            if n_cases_class_1 > 0
            else 0.0
        )
        prevalence_rows.append(
            {
                "activity": activity,
                "translation": resolve_translation(
                    activity, translation_cache, activity_translated_fallback
                ),
                "prevalence_class_0": p0,
                "prevalence_class_1": p1,
                "delta_class_1_minus_class_0": p1 - p0,
            }
        )

    df_prevalence = pd.DataFrame(prevalence_rows)
    df_prevalence = df_prevalence.sort_values(
        by="delta_class_1_minus_class_0",
        key=lambda s: s.abs(),
        ascending=False,
    )
    df_prevalence.to_csv(args.output_dir / "class_action_prevalence_diff.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.hist(actions_per_case.values, bins=50, color="#1f77b4", alpha=0.8)
    plt.axvline(mean_actions, color="red", linestyle="--", label=f"Media: {mean_actions:.2f}")
    plt.axvline(
        median_actions,
        color="green",
        linestyle=":",
        label=f"Mediana: {median_actions:.2f}",
    )
    plt.title("Distribuzione numero azioni per case_id")
    plt.xlabel("Numero di azioni per case")
    plt.ylabel("Numero di case")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "actions_per_case_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    data_for_boxplot = [counts_by_class["class_0"], counts_by_class["class_1"]]
    plt.boxplot(data_for_boxplot, labels=["class_0", "class_1"], showfliers=False)
    plt.title("Quantita di azioni (senza label) per classe")
    plt.ylabel("Numero azioni per case")
    plt.tight_layout()
    plt.savefig(args.output_dir / "class_quantity_boxplot.png", dpi=300)
    plt.close()

    top_diff = df_prevalence.head(min(args.top_k, len(df_prevalence))).copy()
    if not top_diff.empty:
        top_diff = top_diff.sort_values("delta_class_1_minus_class_0", ascending=True)
        plt.figure(figsize=(12, 8))
        plt.barh(
            top_diff["activity"],
            top_diff["delta_class_1_minus_class_0"],
            color=["#d62728" if x < 0 else "#2ca02c" for x in top_diff["delta_class_1_minus_class_0"]],
        )
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Differenza prevalenza azioni per case (class_1 - class_0)")
        plt.xlabel("Delta prevalenza")
        plt.ylabel("Azione")
        plt.tight_layout()
        plt.savefig(args.output_dir / "class_action_prevalence_diff_top.png", dpi=300)
        plt.close()

    summary_lines = [
        "=== DATASET ANALYSIS SUMMARY ===",
        f"Input: {args.input}",
        f"Case unici: {unique_cases}",
        f"Media azioni per case: {mean_actions:.2f}",
        f"Mediana azioni per case: {median_actions:.2f}",
        f"Eventi non-label totali: {total_non_label_events}",
        "",
        "Distribuzione label (ultima azione per case):",
    ]
    for label in ["class_0", "class_1"]:
        summary_lines.append(f"- {label}: {len(class_cases[label])} case")

    summary_lines.extend(
        [
            "",
            "File generati:",
            "- top_actions_overall.csv",
            "- top_actions_class_0.csv",
            "- top_actions_class_1.csv",
            "- class_quantity_summary.csv",
            "- class_action_prevalence_diff.csv",
            "- actions_per_case_distribution.png",
            "- class_quantity_boxplot.png",
            "- class_action_prevalence_diff_top.png (se disponibile)",
        ]
    )

    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(f"Analisi completata. Output in: {args.output_dir}")
    print(f"Case unici: {unique_cases}")
    print(f"Media azioni per case: {mean_actions:.2f}")
    print(f"Mediana azioni per case: {median_actions:.2f}")


if __name__ == "__main__":
    main()