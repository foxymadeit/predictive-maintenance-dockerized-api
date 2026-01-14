from pathlib import Path
import json
import pandas as pd


def load_all_metrics(artifacts_dir: Path) -> pd.DataFrame:
    """Loads all metrics from artifcats directory.

    Args:
        artifacts_dir (Path): Path to the artifacts directory.

    Returns: 
        pd.DataFrame object.
    """
    rows = []

    for model_dir in artifacts_dir.iterdir():
        if not model_dir.is_dir():
            continue

        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        metrics["run_dir"] = model_dir.name
        rows.append(metrics)

    df = pd.DataFrame(rows)

    ORDER = [
        "run_dir",
        "model",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "fp",
        "fn",
        "tp",
        "tn",
        "accuracy",
    ]

    df = df[[c for c in ORDER if c in df.columns]]

    return df


def load_all_threshold_sweeps(artifacts_dir: Path) -> pd.DataFrame:
    """Loads all threshold sweeps from artifacts directory.

    Args:
        artifacts_dir (Path): Path to the artifacts directory.

    Returns:
        pd.DataFrame object.
    """
    dfs = []

    for model_dir in artifacts_dir.iterdir():
        sweep_path = model_dir / "threshold_sweep.csv"
        if not sweep_path.exists():
            continue

        df = pd.read_csv(sweep_path)
        df["run_dir"] = model_dir.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    cols = ["run_dir"] + [c for c in df_all.columns if c != "run_dir"]
    df_all = df_all[cols]

    return df_all
