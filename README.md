# Prediksi Banjir 3 Hari ke Depan — Balikpapan & Samarinda (AutoML)

Memprediksi **kejadian banjir 3 hari ke depan** (`Flood_t+3`) dari data **cuaca harian + banjir**
yang ditransformasi menjadi **time series** (lag feature `t-1..t-7` + akumulasi/rolling),
lalu dilatih dengan **AutoML (FLAML)** + Random Forest / XGBoost / CatBoost.

- **Jenis data:** time series (cuaca berurutan → lag feature).
- **Target:** klasifikasi biner banjir (0/1) di `t+3` → *Time Series Classification*, bukan forecasting nilai kontinu.
- **Lokasi:** `latitude`/`longitude` (titik per kota) dibawa apa adanya ke output prediksi (tidak ditransformasi).
- **Kecamatan (hybrid):** data cuaca tidak punya dimensi kecamatan, jadi kecamatan terdampak dilampirkan ke output dari data historis banjir BPS per kecamatan (prior lokasi). Output prediksi = tanggal + kota + kecamatan + data.

## Jalankan di Google Colab (disarankan)

```python
!git clone <URL-REPO-GITHUB-MU>.git
%cd machine-learning
```

Lalu buka **`Flood_AutoML_Tabular.ipynb`** dan jalankan sel dari atas ke bawah.
Sel pertama memasang dependensi otomatis saat di Colab. Notebook memilih `MODE` sendiri:

| Data tersedia | `MODE` | Hasil |
|---|---|---|
| **DEM + RBI tidak ada** (default repo ini) | `timeseries` | Prediksi banjir `t+3` dari cuaca + banjir. |
| **DEM + RBI ada** (extract zip ke `dem/`, `RBI/`, `padat-penduduk/`) | `geospatial` | Peta probabilitas banjir per kota. |

## Jalankan lokal

```bash
pip install -r requirements.txt
jupyter notebook Flood_AutoML_Tabular.ipynb
```

## Struktur repo

```
machine-learning/
├─ Flood_AutoML_Tabular.ipynb     # NOTEBOOK UTAMA (mode otomatis)
├─ requirements.txt
├─ intruksi.md                    # catatan revisi dosen
├─ dataset/                       # data mentah
│  ├─ data-cuaca_smdbpp16-26.csv  #   cuaca harian (Open-Meteo)
│  └─ bnpb-banjir_Smdbpp.csv      #   kejadian banjir berlabel tanggal (BNPB)
├─ processed/
│  ├─ build_primary_dataset.py    # mentah  -> dataset_utama.csv (cuaca + Flood harian)
│  ├─ build_timeseries_dataset.py # utama   -> dataset_timeseries.csv (lag + rolling + Flood_t+3)
│  ├─ dataset_utama.csv
│  └─ dataset_timeseries.csv      # dataset siap latih (sudah disertakan)
├─ clean/                         # data wilayah (untuk fitur wilayah & mode geospasial)
│  ├─ banjir_balikpapan.csv  banjir_samarinda.csv     # banjir per kecamatan
│  ├─ penduduk_*.csv                                  # kepadatan penduduk
│  └─ admin/*.geojson                                 # batas kecamatan/kelurahan
├─ banjir/                        # sumber mentah banjir per kecamatan (BPS)
└─ scripts/build_notebooks.py     # generator notebook (regenerasi bila perlu)
```

## Membangun ulang data & notebook

```bash
python processed/build_primary_dataset.py       # -> processed/dataset_utama.csv
python processed/build_timeseries_dataset.py    # -> processed/dataset_timeseries.csv
python scripts/build_notebooks.py               # -> Flood_AutoML_Tabular.ipynb
```

## Transformasi time series (inti revisi dosen)

- Untuk tiap variabel cuaca (`precip, tmax, tmin, wind, rain, soil`) dibuat **lag `t-1..t-7`**.
- Ditambah **akumulasi**: hujan 3/7/14 hari, presipitasi 3/7 hari, kelembapan tanah rata-rata 3/7 hari.
- **Target** = banjir pada `t+3` (geser −3 per kota).
- **Anti-leakage:** fitur hanya memakai data ≤ `t`; split **berbasis waktu** (70% awal / 30% akhir);
  imbalance (SMOTE / class weight) ditangani **hanya pada data latih, setelah split**.
- `latitude`/`longitude` **tidak diolah** — hanya dibawa sebagai info lokasi ke output.
- **Output** = tanggal, kota, kecamatan (dari historis BPS), jumlah banjir historis, koordinat kota, probabilitas.
- Metrik utama: **Recall** & **AUC-PR** (lebih relevan dari accuracy untuk data sangat timpang;
  banjir ≈ 0,8% dari data).
