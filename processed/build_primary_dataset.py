from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEATHER_PATH = ROOT / "dataset" / "data-cuaca_smdbpp16-26.csv"
BNPB_PATH = ROOT / "dataset" / "bnpb-banjir_Smdbpp.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset_utama.csv"

LOCATION_MAP = {
    0: "Kota Balikpapan",
    1: "Kota Samarinda",
}


def load_weather(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    location_meta = pd.read_csv(path, nrows=2)
    weather = pd.read_csv(path, skiprows=4)

    location_meta["city"] = location_meta["location_id"].map(LOCATION_MAP)
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce").dt.normalize()
    weather["city"] = weather["location_id"].map(LOCATION_MAP)

    if location_meta["city"].isna().any() or weather["city"].isna().any():
        raise ValueError("Ada location_id cuaca yang belum dipetakan ke kota.")

    return location_meta, weather


def load_bnpb(path: Path) -> pd.DataFrame:
    bnpb = pd.read_csv(path)
    bnpb["Tanggal / Waktu Kejadian"] = pd.to_datetime(
        bnpb["Tanggal / Waktu Kejadian"], errors="coerce"
    ).dt.normalize()
    bnpb = bnpb[bnpb["Kabupaten"].isin(LOCATION_MAP.values())].copy()

    labels = (
        bnpb.groupby(["Kabupaten", "Tanggal / Waktu Kejadian"], as_index=False)
        .agg(Flood=("is_bencana", "max"))
        .rename(columns={"Kabupaten": "city", "Tanggal / Waktu Kejadian": "time"})
    )
    labels["Flood"] = labels["Flood"].astype(int)
    return labels


def build_primary_dataset() -> pd.DataFrame:
    location_meta, weather = load_weather(WEATHER_PATH)
    labels = load_bnpb(BNPB_PATH)

    weather = weather.merge(location_meta, on=["location_id", "city"], how="left")

    min_label_date = labels["time"].min()
    max_label_date = labels["time"].max()
    weather = weather[(weather["time"] >= min_label_date) & (weather["time"] <= max_label_date)].copy()

    dataset = weather.merge(labels, on=["city", "time"], how="left")
    dataset["Flood"] = dataset["Flood"].fillna(0).astype(int)
    dataset = dataset[
        [
            "city",
            "location_id",
            "latitude",
            "longitude",
            "time",
            "precipitation_sum (mm)",
            "temperature_2m_max (°C)",
            "temperature_2m_min (°C)",
            "wind_speed_10m_max (km/h)",
            "rain_sum (mm)",
            "soil_moisture_0_to_100cm_mean (m³/m³)",
            "Flood",
        ]
    ].sort_values(["city", "time"]).reset_index(drop=True)

    return dataset


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_primary_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"saved: {OUTPUT_PATH}")
    print(f"rows: {len(dataset)}")
    print(f"date range: {dataset['time'].min()} -> {dataset['time'].max()}")
    print(dataset.groupby('city')['Flood'].agg(['count', 'sum']).to_string())


if __name__ == "__main__":
    main()