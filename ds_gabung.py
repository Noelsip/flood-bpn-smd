import pandas as pd

bnpb = pd.read_csv("dataset/bnpb-banjir_Smdbpp.csv")
weather = pd.read_csv("dataset/data-cuaca_smdbpp16-26.csv")

weather["time"] = pd.to_datetime(weather["time"])
bnpb["Tanggal / Waktu Kejadian"] = pd.to_datetime(
    bnpb["Tanggal / Waktu Kejadian"]
)

weather["Flood"] = weather["time"].isin(
    bnpb["Tanggal / Waktu Kejadian"]
).astype(int)

print(weather.head())