"""
Build one central, model-ready time-series dataset.

Output:
    processed/dataset_timeseries.csv

Split policy:
    - test       = Balikpapan/Samarinda rows with time >= 2023-01-01
    - validation = full Balikpapan/Samarinda rows from 2021-01-01 to 2022-12-31
    - train = all external-city rows from all available years + Balikpapan/Samarinda before 2023

Training negatives are downsampled to 5:1 (negative:positive): up to 2:1 from
the 21 days before flood-target windows, then the rest from normal/background
non-flood days.

Target:
    Flood_next_3d = 1 if flood occurs any time in the next 1..3 days.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
PRIMARY_PATH = PROCESSED_DIR / "dataset_utama.csv"
OUTPUT_PATH = PROCESSED_DIR / "dataset_timeseries.csv"
DATASET_DIR = ROOT / "dataset"
FLOOD_DIR = ROOT / "banjir"

HORIZON = 3
TARGET_COL = "Flood_next_3d"
LAGS = 7
SEED = 42
SPLIT_CUTOFF_DATE = "2023-01-01"
VALIDATION_START_DATE = "2021-01-01"
UNDERSAMPLE_RATIO = 5
PREFLOOD_NEG_RATIO = 2
NEGATIVE_LOOKBACK_DAYS = 21
EASY_NEG_PERCENTILE = 50
KALTIM_CITY_NAMES = ["Kota Balikpapan", "Kota Samarinda"]
EXTERNAL_FLOOD_DATECOL = "Tanggal / Waktu Kejadian"
EXTERNAL_CITIES = {
    "Bandung": dict(weather=DATASET_DIR / "cuaca bandung 16-26.csv", flood=FLOOD_DIR / "bandung-210 banjir.csv"),
    "Bekasi": dict(weather=DATASET_DIR / "cuaca bekasi 16-26.csv", flood=FLOOD_DIR / "bekasi-134 banjir.csv"),
    "Bogor": dict(weather=DATASET_DIR / "cuaca bogor 16-26.csv", flood=FLOOD_DIR / "bogor-352 banjir.csv"),
    "Pasuruan": dict(weather=DATASET_DIR / "cuaca pasuruan 16-26.csv", flood=FLOOD_DIR / "pasuruan - 154 banjir.csv"),
}

SHORT = ["precip", "tmax", "tmin", "wind", "rain", "soil"]
LAG_COLS = [f"{c}_lag{k}" for c in SHORT for k in range(1, LAGS + 1)]
ROLL_COLS = [
    "rain_roll3_sum", "rain_roll7_sum", "rain_roll14_sum",
    "precip_roll3_sum", "precip_roll7_sum",
    "soil_roll3_mean", "soil_roll7_mean", "tmax_roll7_mean",
]
PCTL_COLS = ["rain7_pctl", "precip7_pctl", "soil7_pctl"]
CALENDAR_COLS = ["doy_sin", "doy_cos", "month_sin", "month_cos"]
FEATURES = LAG_COLS + ROLL_COLS + PCTL_COLS + CALENDAR_COLS


def standardize_weather_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low.startswith("precipitation_sum"):
            rename[col] = "precip"
        elif low.startswith("temperature_2m_max"):
            rename[col] = "tmax"
        elif low.startswith("temperature_2m_min"):
            rename[col] = "tmin"
        elif low.startswith("wind_speed_10m_max"):
            rename[col] = "wind"
        elif low.startswith("rain_sum"):
            rename[col] = "rain"
        elif low.startswith("soil_moisture_0_to_100cm_mean"):
            rename[col] = "soil"
    return df.rename(columns=rename)


def load_kaltim_daily() -> pd.DataFrame:
    if not PRIMARY_PATH.exists():
        raise FileNotFoundError(f"{PRIMARY_PATH} tidak ada. Jalankan build_primary_dataset.py dulu.")
    df = standardize_weather_cols(pd.read_csv(PRIMARY_PATH))
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.normalize()
    return df[["city", "time", *SHORT, "Flood"]].copy()


def load_external_daily(name: str, cfg: dict) -> pd.DataFrame | None:
    wpath, fpath = Path(cfg["weather"]), Path(cfg["flood"])
    if not wpath.exists() or not fpath.exists():
        print(f"[SKIP] {name}: file tidak ditemukan")
        return None

    lines = wpath.read_text(encoding="utf-8").splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.lstrip().startswith("time,"))
    except StopIteration:
        print(f"[SKIP] {name}: header time tidak ditemukan")
        return None

    weather = standardize_weather_cols(pd.read_csv(wpath, skiprows=header_idx))
    missing = [c for c in SHORT if c not in weather.columns]
    if missing:
        print(f"[SKIP] {name}: kolom cuaca hilang {missing}")
        return None

    weather["time"] = pd.to_datetime(weather["time"], errors="coerce").dt.normalize()
    weather = weather.dropna(subset=["time"]).sort_values("time")
    full = pd.date_range(weather["time"].min(), weather["time"].max(), freq="D")
    weather = weather.set_index("time").reindex(full)[SHORT].rename_axis("time").reset_index()

    flood = pd.read_csv(fpath)
    if "date" in flood.columns:
        dates = pd.to_datetime(flood["date"], errors="coerce").dt.normalize()
        flood_set = set(dates.dropna())
    elif {"start_date", "end_date"}.issubset(flood.columns):
        flood_set = set()
        starts = pd.to_datetime(flood["start_date"], errors="coerce").dt.normalize()
        ends = pd.to_datetime(flood["end_date"], errors="coerce").dt.normalize()
        for start, end in zip(starts, ends):
            if pd.notna(start) and pd.notna(end) and end >= start:
                flood_set.update(pd.date_range(start, end, freq="D"))
    elif EXTERNAL_FLOOD_DATECOL in flood.columns:
        dates = pd.to_datetime(flood[EXTERNAL_FLOOD_DATECOL], errors="coerce").dt.normalize()
        flood_set = set(dates.dropna())
    else:
        print(f"[SKIP] {name}: kolom tanggal banjir tidak ditemukan")
        return None

    weather["Flood"] = weather["time"].isin(flood_set).astype(int)
    weather["city"] = name
    return weather[["city", "time", *SHORT, "Flood"]]


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["city", "time"]).reset_index(drop=True)
    grouped = df.groupby("city", sort=False)
    day_of_year = df["time"].dt.dayofyear.astype(float)
    month = df["time"].dt.month.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    for col in SHORT:
        for lag in range(1, LAGS + 1):
            df[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    def roll(col: str, window: int, how: str) -> pd.Series:
        shifted = grouped[col].shift(1)
        per_city = shifted.groupby(df["city"], sort=False)
        if how == "sum":
            return per_city.transform(lambda s: s.rolling(window, min_periods=window).sum())
        return per_city.transform(lambda s: s.rolling(window, min_periods=window).mean())

    df["rain_roll3_sum"] = roll("rain", 3, "sum")
    df["rain_roll7_sum"] = roll("rain", 7, "sum")
    df["rain_roll14_sum"] = roll("rain", 14, "sum")
    df["precip_roll3_sum"] = roll("precip", 3, "sum")
    df["precip_roll7_sum"] = roll("precip", 7, "sum")
    df["soil_roll3_mean"] = roll("soil", 3, "mean")
    df["soil_roll7_mean"] = roll("soil", 7, "mean")
    df["tmax_roll7_mean"] = roll("tmax", 7, "mean")

    def expanding_percentile(series: pd.Series) -> pd.Series:
        return series.expanding(min_periods=10).rank(pct=True)

    df["rain7_pctl"] = df.groupby("city")["rain_roll7_sum"].transform(expanding_percentile)
    df["precip7_pctl"] = df.groupby("city")["precip_roll7_sum"].transform(expanding_percentile)
    df["soil7_pctl"] = df.groupby("city")["soil_roll7_mean"].transform(expanding_percentile)

    target = TARGET_COL
    future = [grouped["Flood"].shift(-step) for step in range(1, HORIZON + 1)]
    df[target] = pd.concat(future, axis=1).max(axis=1)
    df = df.dropna(subset=FEATURES + [target]).reset_index(drop=True)
    df[target] = df[target].astype(int)
    return df


def select_balanced_train(train_pool: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    rng = np.random.RandomState(SEED)
    y = train_pool[target].astype(int).to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    target_neg = UNDERSAMPLE_RATIO * len(pos_idx)
    target_preflood = min(PREFLOOD_NEG_RATIO * len(pos_idx), target_neg)

    preflood_mask = np.zeros(len(train_pool), dtype=bool)
    for _, group in train_pool.groupby("city", sort=False):
        pos_times = pd.to_datetime(group.loc[group[target] == 1, "time"])
        group_idx = group.index.to_numpy()
        group_times = pd.to_datetime(group["time"])
        group_mask = np.zeros(len(group), dtype=bool)
        for flood_time in pos_times:
            start = flood_time - pd.Timedelta(days=NEGATIVE_LOOKBACK_DAYS)
            group_mask |= ((group_times >= start) & (group_times < flood_time)).to_numpy()
        preflood_mask[group_idx] = group_mask

    preflood_neg = neg_idx[preflood_mask[neg_idx]]
    hard_mask = train_pool["rain7_pctl"].to_numpy() >= (EASY_NEG_PERCENTILE / 100.0)
    hard_neg = np.setdiff1d(neg_idx[hard_mask[neg_idx]], preflood_neg, assume_unique=False)
    easy_neg = np.setdiff1d(neg_idx[~hard_mask[neg_idx]], preflood_neg, assume_unique=False)

    selected = []
    if len(preflood_neg):
        selected.extend(rng.choice(preflood_neg, size=min(target_preflood, len(preflood_neg)), replace=False).tolist())
    need = target_neg - len(selected)
    if need > 0 and len(hard_neg):
        selected.extend(rng.choice(hard_neg, size=min(need, len(hard_neg)), replace=False).tolist())
    need = target_neg - len(selected)
    if need > 0 and len(easy_neg):
        selected.extend(rng.choice(easy_neg, size=min(need, len(easy_neg)), replace=False).tolist())

    keep_neg = np.array(selected, dtype=int)
    keep = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep)
    stats = {
        "pos": len(pos_idx), "neg_before": len(neg_idx), "neg_after": len(keep_neg),
        "preflood": int(np.isin(keep_neg, preflood_neg).sum()),
        "hard": int(np.isin(keep_neg, hard_neg).sum()),
        "easy": int(np.isin(keep_neg, easy_neg).sum()),
    }
    return train_pool.iloc[keep].copy().reset_index(drop=True), stats


def build() -> pd.DataFrame:
    daily_frames = [load_kaltim_daily()]
    for city, cfg in EXTERNAL_CITIES.items():
        frame = load_external_daily(city, cfg)
        if frame is not None:
            daily_frames.append(frame)

    daily = pd.concat(daily_frames, ignore_index=True).sort_values(["city", "time"]).reset_index(drop=True)
    ts = make_features(daily)
    target = TARGET_COL
    cutoff = pd.Timestamp(SPLIT_CUTOFF_DATE)
    is_kaltim = ts["city"].isin(KALTIM_CITY_NAMES)

    validation_start = pd.Timestamp(VALIDATION_START_DATE)
    is_validation = is_kaltim & (ts["time"] >= validation_start) & (ts["time"] < cutoff)
    train_pool = ts[(~is_kaltim) | (ts["time"] < validation_start)].copy().reset_index(drop=True)
    validation = ts[is_validation].copy().reset_index(drop=True)
    test = ts[is_kaltim & (ts["time"] >= cutoff)].copy().reset_index(drop=True)
    train, stats = select_balanced_train(train_pool, target)
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"
    out = pd.concat([train, validation, test], ignore_index=True).sort_values(["split", "time", "city"]).reset_index(drop=True)
    print("balance train:", stats)
    return out


def main() -> None:
    df = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    target = TARGET_COL
    print(f"saved : {OUTPUT_PATH}")
    print(f"rows  : {len(df)} | cols: {df.shape[1]}")
    print(df.groupby(["split", "city"])[target].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
