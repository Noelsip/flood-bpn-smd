from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


SEED = 42
TARGET = "Flood_next_3d"
DEFAULT_CITIES = ["Kota Balikpapan", "Kota Samarinda"]
DEFAULT_CONTAMINATION = 0.05
DEFAULT_SPLIT_MODE = "chronological_per_city"
DEFAULT_MAX_KECAMATAN_PER_ALERT = 3
DEFAULT_OPERATIONAL_TOP_RATE = 0.05
CALENDAR_FEATURES = {"doy_sin", "doy_cos", "month_sin", "month_cos"}
PERCENTILE_FEATURES = {"rain7_pctl", "precip7_pctl", "soil7_pctl"}


@dataclass(frozen=True)
class CityIsolationResult:
    city: str
    train: pd.DataFrame
    test: pd.DataFrame
    summary: dict
    model: IsolationForest
    scaler: StandardScaler


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric hydrometeorological features for Isolation Forest fitting."""
    excluded = {"Flood", TARGET, "split", *CALENDAR_FEATURES, *PERCENTILE_FEATURES}
    return [
        col
        for col in data.select_dtypes(include=[np.number]).columns
        if col not in excluded and not col.endswith("_pctl")
    ]


def add_rank_anomaly_flags(
    frame: pd.DataFrame,
    *,
    score_col: str = "anomaly_score",
    top_rates: tuple[float, ...] = (0.01, 0.05, 0.10),
    operational_top_rate: float = DEFAULT_OPERATIONAL_TOP_RATE,
) -> pd.DataFrame:
    """Add score-rank flags; the operational anomaly flag is the top score slice."""
    out = frame.sort_values(score_col, ascending=False).copy()
    out["anomaly_rank"] = np.arange(1, len(out) + 1)
    out["anomaly_score_percentile"] = 1 - ((out["anomaly_rank"] - 1) / max(len(out), 1))
    for rate in top_rates:
        pct = int(round(rate * 100))
        n_top = max(1, int(np.ceil(len(out) * rate)))
        out[f"top_{pct}pct_anomaly"] = (out["anomaly_rank"] <= n_top).astype(int)

    operational_pct = int(round(operational_top_rate * 100))
    operational_col = f"top_{operational_pct}pct_anomaly"
    if operational_col not in out.columns:
        n_top = max(1, int(np.ceil(len(out) * operational_top_rate)))
        out[operational_col] = (out["anomaly_rank"] <= n_top).astype(int)
    out["is_anomaly"] = out[operational_col].astype(int)
    out["flood_anomaly"] = out["is_anomaly"]
    return out.sort_values("time").reset_index(drop=True)


def split_city_data(
    city_df: pd.DataFrame,
    *,
    train_ratio: float = 0.80,
    split_mode: str = DEFAULT_SPLIT_MODE,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one city into train/test while keeping the model city-specific."""
    ordered = city_df.sort_values("time").reset_index(drop=True)
    if split_mode == "stratified_label_per_city" and TARGET in ordered.columns:
        rng = np.random.RandomState(random_state)
        train_parts = []
        test_parts = []
        for _, group in ordered.groupby(TARGET, sort=True):
            idx = group.index.to_numpy()
            rng.shuffle(idx)
            n_train = int(np.floor(len(idx) * train_ratio))
            if len(idx) > 1:
                n_train = min(max(n_train, 1), len(idx) - 1)
            train_parts.append(ordered.loc[idx[:n_train]])
            test_parts.append(ordered.loc[idx[n_train:]])
        train_df = pd.concat(train_parts, ignore_index=True).sort_values("time")
        test_df = pd.concat(test_parts, ignore_index=True).sort_values("time")
    elif split_mode == "existing_split" and "split" in ordered.columns and {"train", "test"}.issubset(set(ordered["split"])):
        train_df = ordered[ordered["split"] == "train"].copy()
        test_df = ordered[ordered["split"] == "test"].copy()
    else:
        cut = int(len(ordered) * train_ratio)
        train_df = ordered.iloc[:cut].copy()
        test_df = ordered.iloc[cut:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Train/test kosong untuk kota {ordered['city'].iloc[0]!r}.")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_isolation_forest_for_city(
    city_df: pd.DataFrame,
    features: list[str],
    *,
    contamination: float = DEFAULT_CONTAMINATION,
    train_ratio: float = 0.80,
    split_mode: str = DEFAULT_SPLIT_MODE,
    operational_top_rate: float = DEFAULT_OPERATIONAL_TOP_RATE,
    random_state: int = SEED,
) -> CityIsolationResult:
    """Fit scaler and Isolation Forest on one city's train fold, then score its test fold."""
    city = str(city_df["city"].iloc[0])
    train_df, test_df = split_city_data(
        city_df,
        train_ratio=train_ratio,
        split_mode=split_mode,
        random_state=random_state,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[features])
    x_test = scaler.transform(test_df[features])

    model = IsolationForest(
        n_estimators=400,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train)

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out["anomaly_score"] = -model.score_samples(x_train)
    test_out["anomaly_score"] = -model.score_samples(x_test)
    train_out["model_is_anomaly"] = (model.predict(x_train) == -1).astype(int)
    test_out["model_is_anomaly"] = (model.predict(x_test) == -1).astype(int)
    train_out = add_rank_anomaly_flags(train_out, operational_top_rate=operational_top_rate)
    test_out = add_rank_anomaly_flags(test_out, operational_top_rate=operational_top_rate)
    train_out["anomaly_interpretation"] = np.where(
        train_out["is_anomaly"].eq(1),
        "anomali cuaca",
        "normal",
    )
    test_out["anomaly_interpretation"] = np.where(
        test_out["is_anomaly"].eq(1),
        "anomali cuaca",
        "normal",
    )

    historical_flood_days = int(test_out[TARGET].sum()) if TARGET in test_out.columns else np.nan
    anomaly_on_historical_flood = (
        int(((test_out["is_anomaly"] == 1) & (test_out[TARGET] == 1)).sum())
        if TARGET in test_out.columns
        else np.nan
    )
    model_anomaly_on_historical_flood = (
        int(((test_out["model_is_anomaly"] == 1) & (test_out[TARGET] == 1)).sum())
        if TARGET in test_out.columns
        else np.nan
    )
    model_detected_anomaly = int(test_out["model_is_anomaly"].sum())
    summary = {
        "city": city,
        "split_mode": split_mode,
        "operational_top_rate": operational_top_rate,
        "train_days": int(len(train_out)),
        "test_days": int(len(test_out)),
        "train_start": train_out["time"].min(),
        "train_end": train_out["time"].max(),
        "test_start": test_out["time"].min(),
        "test_end": test_out["time"].max(),
        "detected_anomaly": int(test_out["is_anomaly"].sum()),
        "anomaly_rate": float(test_out["is_anomaly"].mean()),
        "model_detected_anomaly": model_detected_anomaly,
        "model_anomaly_rate": float(test_out["model_is_anomaly"].mean()),
        "mean_anomaly_score": float(test_out["anomaly_score"].mean()),
        "max_anomaly_score": float(test_out["anomaly_score"].max()),
        "historical_flood_days": historical_flood_days,
        "anomaly_on_historical_flood": anomaly_on_historical_flood,
        "model_anomaly_on_historical_flood": model_anomaly_on_historical_flood,
    }
    return CityIsolationResult(city, train_out, test_out, summary, model, scaler)


def run_per_city_isolation_forest(
    data: pd.DataFrame,
    *,
    cities: list[str] | None = None,
    contamination: float = DEFAULT_CONTAMINATION,
    train_ratio: float = 0.80,
    split_mode: str = DEFAULT_SPLIT_MODE,
    operational_top_rate: float = DEFAULT_OPERATIONAL_TOP_RATE,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, CityIsolationResult]]:
    """Run an independent Isolation Forest pipeline for each requested city."""
    work = data.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    cities = cities or DEFAULT_CITIES
    work = work[work["city"].isin(cities)].dropna(subset=["time"]).copy()
    features = feature_columns(work)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    results: dict[str, CityIsolationResult] = {}
    for city in cities:
        city_df = work[work["city"] == city].copy()
        if city_df.empty:
            raise ValueError(f"Data kota {city!r} tidak ditemukan.")
        results[city] = fit_isolation_forest_for_city(
            city_df,
            features,
            contamination=contamination,
            train_ratio=train_ratio,
            split_mode=split_mode,
            operational_top_rate=operational_top_rate,
            random_state=random_state,
        )

    summary = pd.DataFrame([result.summary for result in results.values()])
    predictions = pd.concat([result.test for result in results.values()], ignore_index=True)
    predictions = predictions.sort_values(["city", "time", "anomaly_score"], ascending=[True, True, False])
    return summary, predictions, results


def load_kecamatan_prior(clean_dir: Path, *, max_per_city: int = DEFAULT_MAX_KECAMATAN_PER_ALERT) -> pd.DataFrame:
    """Load most flood-prone subdistricts per city for output-only location context."""
    frames = []
    for path in [clean_dir / "banjir_balikpapan.csv", clean_dir / "banjir_samarinda.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=["city", "kecamatan", "banjir_historis"])

    prior = pd.concat(frames, ignore_index=True)
    prior = prior.rename(columns={"kota": "city", "banjir_count": "banjir_historis"})
    needed = ["city", "kecamatan", "banjir_historis"]
    prior = prior[[col for col in needed if col in prior.columns]].copy()
    if not set(needed).issubset(prior.columns):
        return pd.DataFrame(columns=needed)

    prior["banjir_historis"] = pd.to_numeric(prior["banjir_historis"], errors="coerce").fillna(0)
    prior = prior.sort_values(["city", "banjir_historis", "kecamatan"], ascending=[True, False, True])
    return prior.groupby("city", as_index=False).head(max_per_city).reset_index(drop=True)


def attach_kecamatan_prior(
    predictions: pd.DataFrame,
    clean_dir: Path,
    *,
    max_per_city: int = DEFAULT_MAX_KECAMATAN_PER_ALERT,
    anomalies_only: bool = True,
) -> pd.DataFrame:
    """Attach output-only kecamatan priors to anomaly rows."""
    prior = load_kecamatan_prior(clean_dir, max_per_city=max_per_city)
    base = predictions.copy()
    if anomalies_only and "is_anomaly" in base.columns:
        base = base[base["is_anomaly"] == 1].copy()
    if prior.empty:
        base["kecamatan"] = np.nan
        base["banjir_historis"] = np.nan
        return base
    return base.merge(prior, on="city", how="left")


def evaluate_anomaly_overlap(
    predictions: pd.DataFrame,
    *,
    target: str = TARGET,
    top_rates: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Post-hoc overlap and ranking analysis against historical flood records."""
    rows = []
    if target not in predictions.columns:
        return pd.DataFrame()

    for city, group in predictions.groupby("city", sort=True):
        frame = group.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        flood = frame[target].astype(int)
        anomaly = frame["is_anomaly"].astype(int)
        historical_flood_days = int(flood.sum())
        anomaly_days = int(anomaly.sum())
        flood_days_flagged = int(((anomaly == 1) & (flood == 1)).sum())
        row = {
            "city": city,
            "test_days": int(len(frame)),
            "historical_flood_days": historical_flood_days,
            "anomaly_days": anomaly_days,
            "anomaly_rate": float(anomaly.mean()),
            "flood_days_flagged": flood_days_flagged,
            "overlap_rate": (
                flood_days_flagged / historical_flood_days
                if historical_flood_days
                else np.nan
            ),
            "mean_score_flood_days": (
                float(frame.loc[flood == 1, "anomaly_score"].mean())
                if historical_flood_days
                else np.nan
            ),
            "mean_score_non_flood_days": (
                float(frame.loc[flood == 0, "anomaly_score"].mean())
                if int((flood == 0).sum())
                else np.nan
            ),
        }
        for rate in top_rates:
            n_top = max(1, int(np.ceil(len(frame) * rate)))
            top = frame.head(n_top)
            hits = int(top[target].astype(int).sum())
            pct = int(round(rate * 100))
            row[f"top_{pct}pct_days"] = n_top
            row[f"top_{pct}pct_flood_hits"] = hits
            row[f"top_{pct}pct_hit_rate"] = (
                hits / historical_flood_days
                if historical_flood_days
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    kecamatan_predictions: pd.DataFrame | None = None,
    evaluation: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "isolation_forest_summary_per_city.csv", index=False)
    predictions.to_csv(output_dir / "isolation_forest_anomaly_all.csv", index=False)
    if kecamatan_predictions is not None:
        kecamatan_predictions.to_csv(output_dir / "isolation_forest_anomaly_kecamatan.csv", index=False)
    if evaluation is not None:
        evaluation.to_csv(output_dir / "isolation_forest_overlap_evaluation.csv", index=False)
    for city, group in predictions.groupby("city"):
        slug = city.lower().replace("kota ", "").replace(" ", "_")
        group.to_csv(output_dir / f"isolation_forest_anomaly_{slug}.csv", index=False)
