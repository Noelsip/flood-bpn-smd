"""
build_timeseries_dataset.py
===========================
Mengubah dataset harian `dataset_utama.csv` (snapshot cuaca + Flood 0/1) menjadi
dataset TIME SERIES untuk klasifikasi banjir (Time Series Classification).

Sesuai revisi dosen:
  - Data disusun temporal (urut per kota + tanggal).
  - Dibuat lag feature t-1 .. t-7 untuk tiap variabel cuaca.
  - Ditambah rolling/agregasi (akumulasi hujan 3/7/14 hari, dst).
  - Target = Flood pada t+HORIZON (default 3 hari ke depan -> Flood_t+3).
  - Tidak ada kebocoran masa depan: fitur hanya memakai data <= t, target di t+3.

Kolom `latitude` & `longitude` (titik lokasi per kota) dibiarkan apa adanya, tidak
ditransformasi. Kolom ini dipakai sebagai info lokasi & dibawa ke output prediksi
agar hasil prediksi menyertakan koordinat.

Output: processed/dataset_timeseries.csv
Jalankan:  python processed/build_timeseries_dataset.py
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_PATH = ROOT / "processed" / "dataset_utama.csv"
OUTPUT_PATH = ROOT / "processed" / "dataset_timeseries.csv"

# Horizon peramalan: prediksi banjir berapa hari ke depan.
FORECAST_HORIZON = 3   # Flood_t+3

# Jendela observasi lag (t-1 .. t-LAGS).
LAGS = 7

# Variabel cuaca mentah -> nama pendek untuk fitur lag.
WEATHER_COLS = {
    "precipitation_sum (mm)": "precip",
    "temperature_2m_max (°C)": "tmax",
    "temperature_2m_min (°C)": "tmin",
    "wind_speed_10m_max (km/h)": "wind",
    "rain_sum (mm)": "rain",
    "soil_moisture_0_to_100cm_mean (m³/m³)": "soil",
}


def load_primary() -> pd.DataFrame:
    if not PRIMARY_PATH.exists():
        raise FileNotFoundError(
            f"{PRIMARY_PATH} tidak ada. Jalankan dulu processed/build_primary_dataset.py."
        )
    df = pd.read_csv(PRIMARY_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.rename(columns=WEATHER_COLS)
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["time"].dt.month
    df["dayofyear"] = df["time"].dt.dayofyear
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city", sort=False)
    short_names = list(WEATHER_COLS.values())
    for col in short_names:
        for k in range(1, LAGS + 1):
            df[f"{col}_lag{k}"] = g[col].shift(k)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi akumulasi. Pakai shift(1) lalu rolling agar hanya memakai
    data SAMPAI t-1 (aman, tidak membocorkan hari t maupun masa depan)."""
    g = df.groupby("city", sort=False)

    def roll(col, window, how):
        s = g[col].shift(1)
        r = s.rolling(window, min_periods=window)
        return r.sum() if how == "sum" else r.mean()

    df["rain_roll3_sum"] = roll("rain", 3, "sum")
    df["rain_roll7_sum"] = roll("rain", 7, "sum")
    df["rain_roll14_sum"] = roll("rain", 14, "sum")
    df["precip_roll3_sum"] = roll("precip", 3, "sum")
    df["precip_roll7_sum"] = roll("precip", 7, "sum")
    df["soil_roll3_mean"] = roll("soil", 3, "mean")
    df["soil_roll7_mean"] = roll("soil", 7, "mean")
    df["tmax_roll7_mean"] = roll("tmax", 7, "mean")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city", sort=False)
    df[f"Flood_t+{FORECAST_HORIZON}"] = g["Flood"].shift(-FORECAST_HORIZON)
    return df


def build() -> pd.DataFrame:
    df = load_primary()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    target_col = f"Flood_t+{FORECAST_HORIZON}"
    feature_cols = [c for c in df.columns if c.endswith(tuple(
        [f"_lag{k}" for k in range(1, LAGS + 1)]
    )) or c.endswith(("_sum", "_mean"))]

    # Buang baris yang fiturnya belum lengkap (awal seri) atau target kosong (akhir seri).
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    df[target_col] = df[target_col].astype(int)
    return df


def main() -> None:
    df = build()
    df.to_csv(OUTPUT_PATH, index=False)
    target_col = f"Flood_t+{FORECAST_HORIZON}"

    print(f"saved : {OUTPUT_PATH}")
    print(f"rows  : {len(df)} | cols: {df.shape[1]}")
    print(f"target: {target_col} (prediksi {FORECAST_HORIZON} hari ke depan)")
    print(f"range : {df['time'].min().date()} -> {df['time'].max().date()}")
    print("\ndistribusi target per kota (count, banjir):")
    print(df.groupby("city")[target_col].agg(["count", "sum"]).to_string())
    pos = int(df[target_col].sum())
    print(f"\ntotal banjir(t+{FORECAST_HORIZON}): {pos} / {len(df)} "
          f"({pos / len(df) * 100:.2f}%)")


if __name__ == "__main__":
    main()
