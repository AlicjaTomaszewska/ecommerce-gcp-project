import pandas as pd

def split_by_snapshots(df: pd.DataFrame, date_col: str = "snapshot_date", val_last_n: int = 1):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    snapshots = sorted(df[date_col].dropna().unique())
    if len(snapshots) <= val_last_n:
        raise ValueError(f"Not enough snapshots. Found={len(snapshots)}, val_last_n={val_last_n}")
    val_dates = snapshots[-val_last_n:]
    train_dates = snapshots[:-val_last_n]
    return (
        df[df[date_col].isin(train_dates)].copy(),
        df[df[date_col].isin(val_dates)].copy(),
        train_dates,
        val_dates,
    )
