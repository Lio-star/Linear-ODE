from __future__ import annotations

import pandas as pd

from model_runner import MODEL_REGISTRY, normalize_models


DISPLAY_NAME_MAP = {
    "NumAllGene": "Number of Gene",
    "NumTF": "Number of TF",
    "B": "B (Trajectory)",
    "DataRun": "DataRun",
    "DataSeed": "DataSeed",
    "K_time": "Time points to Calculate Loss in Autodiff",
    "Autodiff_Time": "Autodiff Time (s)",
    "Autodiff_Loss": "Autodiff Loss",
    "OurModel_Time": "OurModel Time (s)",
    "OurModel_Loss": "OurModel Loss",
}

DISPLAY_ORDER = [
    "Index",
    "DataSeed",
    "Number of Gene",
    "Number of TF",
    "B (Trajectory)",
    "DataRun",
    "Time points to Calculate Loss in Autodiff",
    "Autodiff Time (s)",
    "Autodiff Loss",
    "OurModel Time (s)",
    "OurModel Loss",
]


def _safe_int_str(x):
    if pd.isna(x):
        return ""
    try:
        return f"{int(float(x))}"
    except Exception:
        return str(x)


def _safe_float6_str(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _safe_sci_str(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2E}"
    except Exception:
        return str(x)


def _get_raw_ordered_cols(models):
    base_cols = ["NumAllGene", "NumTF", "B", "DataRun", "DataSeed", "K_time"]
    model_cols = []

    for key in ["autodiff", "our_model"]:
        if key in models:
            model_cols.extend(
                [
                    MODEL_REGISTRY[key]["time_col"],
                    MODEL_REGISTRY[key]["loss_col"],
                ]
            )

    return base_cols + model_cols


def _build_rename_map(models):
    rename_map = {
        "NumAllGene": DISPLAY_NAME_MAP["NumAllGene"],
        "NumTF": DISPLAY_NAME_MAP["NumTF"],
        "B": DISPLAY_NAME_MAP["B"],
        "DataRun": DISPLAY_NAME_MAP["DataRun"],
        "DataSeed": DISPLAY_NAME_MAP["DataSeed"],
        "K_time": DISPLAY_NAME_MAP["K_time"],
    }

    if "autodiff" in models:
        rename_map[MODEL_REGISTRY["autodiff"]["time_col"]] = DISPLAY_NAME_MAP["Autodiff_Time"]
        rename_map[MODEL_REGISTRY["autodiff"]["loss_col"]] = DISPLAY_NAME_MAP["Autodiff_Loss"]

    if "our_model" in models:
        rename_map[MODEL_REGISTRY["our_model"]["time_col"]] = DISPLAY_NAME_MAP["OurModel_Time"]
        rename_map[MODEL_REGISTRY["our_model"]["loss_col"]] = DISPLAY_NAME_MAP["OurModel_Loss"]

    return rename_map


def build_summary_table(results, models):
    models = normalize_models(models)
    raw_ordered_cols = _get_raw_ordered_cols(models)

    df = pd.DataFrame(results)
    if df.empty:
        empty_cols = [c for c in DISPLAY_ORDER if c != "Index"]
        return pd.DataFrame(columns=["Index"] + empty_cols)

    df = df[raw_ordered_cols].copy()

    rename_map = _build_rename_map(models)
    df = df.rename(columns=rename_map)

    df.insert(0, "Index", range(len(df)))

    avg_row = {col: "-" for col in df.columns}
    avg_row["Index"] = "AVG_ALL"

    metric_cols = []
    if "autodiff" in models:
        metric_cols.extend(["Autodiff Time (s)", "Autodiff Loss"])
    if "our_model" in models:
        metric_cols.extend(["OurModel Time (s)", "OurModel Loss"])

    for col in metric_cols:
        if col in df.columns:
            avg_row[col] = pd.to_numeric(df[col], errors="coerce").mean()

    df_final = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    existing_cols = [c for c in DISPLAY_ORDER if c in df_final.columns]
    remaining_cols = [c for c in df_final.columns if c not in existing_cols]

    return df_final[existing_cols + remaining_cols]


def style_experiment_table(df):
    df_show = df.copy()

    if "Index" not in df_show.columns:
        df_show.insert(0, "Index", range(len(df_show)))

    int_like_cols = [
        "Index",
        "DataSeed",
        "Number of Gene",
        "Number of TF",
        "B (Trajectory)",
        "DataRun",
        "Time points to Calculate Loss in Autodiff",
    ]

    time_like_cols = [
        "Autodiff Time (s)",
        "OurModel Time (s)",
    ]

    loss_like_cols = [
        "Autodiff Loss",
        "OurModel Loss",
    ]

    format_dict = {}

    for col in int_like_cols:
        if col in df_show.columns:
            format_dict[col] = _safe_int_str

    for col in time_like_cols:
        if col in df_show.columns:
            format_dict[col] = _safe_float6_str

    for col in loss_like_cols:
        if col in df_show.columns:
            format_dict[col] = _safe_sci_str

    def highlight_avg_row(row):
        is_avg = str(row.get("Index", "")) == "AVG_ALL"
        if is_avg:
            return ["font-weight: bold; background-color: #EAEAEA;" for _ in row]
        return ["" for _ in row]

    styled = (
        df_show.style
        .hide(axis="index")
        .format(format_dict)
        .apply(highlight_avg_row, axis=1)
        .set_caption("Experiment Summary Table")
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "16px"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "#D9EAF7"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #999"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("border", "1px solid #BBB"),
                    ("padding", "6px"),
                ],
            },
            {
                "selector": "tr:nth-child(even) td",
                "props": [("background-color", "#F7F7F7")],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "10px 0"),
                    ("font-size", "13px"),
                    ("width", "100%"),
                ],
            },
        ])
    )

    return styled


def print_final_summary_banner():
    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE (all runs) + OVERALL AVG")
    print("=" * 70)


def save_summary_tables(df_final, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "final_summary.csv"
    display_csv_path = output_dir / "final_summary_display.csv"

    df_final.to_csv(csv_path, index=False)

    df_display = df_final.copy()

    for col in df_display.columns:
        if col.endswith("Time (s)"):
            df_display[col] = df_display[col].apply(_safe_float6_str)
        elif col.endswith("Loss"):
            df_display[col] = df_display[col].apply(_safe_sci_str)

    df_display.to_csv(display_csv_path, index=False)

    return csv_path, display_csv_path