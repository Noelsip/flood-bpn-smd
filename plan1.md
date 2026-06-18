# Plan 1 — Spesifikasi Implementasi: Data Pooling Multi-Kota untuk Model Banjir

> **Dokumen ini adalah spesifikasi eksekusi.** Pembaca (manusia atau LLM agent) harus dapat
> mengimplementasikan tanpa menebak. Semua keputusan sudah dipatok dengan nilai **DEFAULT**.
> Ubah nilai DEFAULT hanya jika instruksi pengguna menyatakan lain.
>
> **Tujuan:** menambah jumlah kejadian banjir untuk pelatihan dengan menyertakan data banjir +
> cuaca dari **Bandung, Bekasi, Bogor, Pasuruan**, **tanpa memperluas wilayah penelitian**.
> Output akhir **tetap Balikpapan & Samarinda lengkap dengan kecamatan**.

---

## STATUS: ✅ Terimplementasi di `Flood_AutoML_Tabular.ipynb` (2026-06-18)

Seluruh pipeline diterapkan **inline sebagai sel notebook** (bukan skrip `scripts/*.py` terpisah),
karena dijalankan langsung di notebook/Colab. Ringkasan sel yang diubah:
- **Sel 4 (config):** `EXTERNAL_CITIES`, `POOL_EXTERNAL`, `SPLIT_CUTOFF_DATE="2022-01-01"`, `UNDERSAMPLE_RATIO=3`, `EASY_NEG_PERCENTILE=50`, `USE_SMOTE=False`, `KALTIM_CITY_NAMES`.
- **Sel 5A:** loader Kaltim + kota luar, lag+rolling+**persentil-ekspansi**, `cat_cols=[]`.
- **Sel 8 (split):** cutoff tanggal + filter TEST ke Kaltim.
- **Sel 9 (imbalance):** undersample hard-negative-aware + class weight.
- **Sel 11.5 (baru):** ablation baseline vs pooled → `outputs/ablation_results.csv`.

> ⚠️ **Untuk Colab:** `dataset/cuaca *.csv` dan `banjir/*.csv` masih *untracked* — commit & push
> dulu agar ikut ter-`git clone`. Untuk Jupyter lokal tidak perlu.

---

## 0. Aturan Wajib (Invarian — jangan dilanggar)

1. `city` / provinsi / `latitude` / `longitude` **TIDAK PERNAH** menjadi fitur input model.
2. Output kecamatan **hanya** dari Modul B (DEM/RBI), **hanya** Balikpapan & Samarinda.
3. Metrik yang dilaporkan **hanya** dihitung pada subset test Balikpapan + Samarinda.
4. Semua statistik (mean/std/persentil/scaler) dihitung **tanpa melihat data masa depan**
   (expanding/training-only). Test set tidak pernah memengaruhi statistik.
5. Balancing (undersample/SMOTE/class-weight) **hanya** pada data training.
6. `processed/dataset_utama.csv` (Kaltim existing) **tidak boleh ditimpa**. Output baru = file baru.

---

## 1. Parameter Global (DEFAULT)

```python
HORIZON              = 3          # target Flood_t+3 (DEFAULT; opsi sah: 1, 3, 7)
LAGS                 = 7          # lag t-1..t-7
NORMALIZATION        = "percentile_expanding"   # metode anomali per-kota (lihat §4.3)
SPLIT_STRATEGY       = "global_date_cutoff"     # lihat §4.5
SPLIT_CUTOFF_DATE    = "2022-01-01"  # train: time < cutoff ; test: time >= cutoff
                                     # dipilih agar test Kaltim berisi 23 event banjir (recall stabil)
UNDERSAMPLE_RATIO    = 3          # negatif : positif = 3 : 1 di training (rasio moderat)
EASY_NEG_PERCENTILE  = 50         # negatif "mudah" = rain_roll7_sum < persentil-50 kota tsb
EXTERNAL_CITIES      = ["Bandung", "Bekasi", "Bogor", "Pasuruan"]
KALTIM_CITIES        = ["Balikpapan", "Samarinda"]
SEED                 = 42
```

---

## 2. Kontrak Skema Data Input (WAJIB tepat)

### 2.1 Cuaca kota luar — `raw_external/<kota>/weather.csv`
Sumber: **Open-Meteo Historical/Archive API (reanalysis ERA5-Land)**. Kolom **harus identik** ini:

| Kolom (header persis) | Tipe | Wajib |
|---|---|---|
| `time` | tanggal `YYYY-MM-DD` | ✅ |
| `precipitation_sum (mm)` | float | ✅ |
| `temperature_2m_max (°C)` | float | ✅ |
| `temperature_2m_min (°C)` | float | ✅ |
| `wind_speed_10m_max (km/h)` | float | ✅ |
| `rain_sum (mm)` | float | ✅ |
| `soil_moisture_0_to_100cm_mean (m³/m³)` | float | ✅ |

**Validasi (gagal → kota di-skip, cetak peringatan):** semua kolom di atas ada; `time` parseable;
tidak ada gap tanggal > 1 hari di tengah rentang (gap diisi `NaN`, lalu di-`dropna` setelah lag).

### 2.2 Kejadian banjir kota luar — `raw_external/<kota>/flood_events.csv`
Format **DEFAULT** (dukung 2 bentuk; deteksi otomatis dari header):

- **Bentuk A (tanggal tunggal):** kolom `date` (`YYYY-MM-DD`). 1 baris = 1 hari banjir.
- **Bentuk B (rentang):** kolom `start_date`, `end_date` (`YYYY-MM-DD`). Semua hari dalam
  `[start_date, end_date]` inklusif ditandai banjir.

Kolom lain (kecamatan, korban, dll.) **diabaikan**. Kecamatan kota luar tidak dipakai.

---

## 3. Struktur Folder / Output

```
raw_external/<kota>/weather.csv , flood_events.csv          # input kota luar
processed/dataset_utama.csv                                  # EXISTING Kaltim — read-only
processed/dataset_utama_multikota.csv                        # OUTPUT Tahap 2
processed/dataset_timeseries.csv                             # OUTPUT Tahap 3-5 (siap model)
processed/city_summary.csv                                   # OUTPUT ringkasan per kota
scripts/build_external_daily.py                              # Tahap 1-2
scripts/build_timeseries_multikota.py                        # Tahap 3-5
outputs/ablation_results.csv                                 # OUTPUT Tahap 8
```

---

## 4. Tahapan (deterministik)

### Tahap 1 — `build_external_daily.py`: pelabelan harian kota luar
Untuk tiap `kota` di `EXTERNAL_CITIES`:
1. Baca `weather.csv`; jalankan validasi §2.1. Rename kolom ke alias (`precip,tmax,tmin,wind,rain,soil`).
2. Bentuk kalender harian penuh dari `min(time)` s/d `max(time)` (frekuensi `D`); reindex; gap → `NaN`.
3. Baca `flood_events.csv`; deteksi Bentuk A/B (§2.2); hasilkan himpunan tanggal banjir.
4. `Flood = 1` untuk tanggal di himpunan, selain itu `0` (int).
5. Tambah `city = "<Kota>"`.
6. **Output kolom (urutan tetap):** `city, time, precip, tmax, tmin, wind, rain, soil, Flood`.

### Tahap 2 — gabung multi-kota → `dataset_utama_multikota.csv`
1. Muat Kaltim dari `processed/dataset_utama.csv`, rename kolom cuaca ke alias yang sama,
   pertahankan hanya kolom `city, time, precip, tmax, tmin, wind, rain, soil, Flood`.
2. `concat` Kaltim + semua kota luar (urutan baris: sort `["city","time"]`).
3. Tulis `processed/dataset_utama_multikota.csv`.
4. Tulis `processed/city_summary.csv`: per `city` → `n_days, n_flood, pct_flood, date_min, date_max`.
   Cetak tabel ini ke stdout.

### Tahap 3 — `build_timeseries_multikota.py`: feature engineering
Semua operasi **per `city`** (`groupby("city")`, `sort_values(["city","time"])`).

**4.1 Lag** — untuk tiap var di `[precip,tmax,tmin,wind,rain,soil]`, buat `<var>_lag1..lag7` via `shift(k)`.

**4.2 Rolling** (semua pakai `shift(1)` dulu agar hanya masa lalu):
```
rain_roll3_sum, rain_roll7_sum, rain_roll14_sum
precip_roll3_sum, precip_roll7_sum
soil_roll3_mean, soil_roll7_mean, tmax_roll7_mean
```

**4.3 Anomali per-kota — metode `percentile_expanding` (DEFAULT, anti-leakage):**
Untuk variabel akumulasi hujan, hitung **persentil ekspansi** = posisi nilai hari-t terhadap
SELURUH nilai kota itu **sampai hari t saja** (tidak melihat masa depan):
```
rain7_pctl[t]   = (# hari ≤ rain_roll7_sum[t] dalam riwayat kota s/d t) / (jumlah hari s/d t)
precip7_pctl[t] = idem untuk precip_roll7_sum
soil7_pctl[t]   = idem untuk soil_roll7_mean
```
Implementasi: `expanding().apply(rank)` atau rolling-rank kumulatif per `city`. Nilai ∈ [0,1].
> Ini membuat "hujan ekstrem" punya arti sama lintas kota tanpa membocorkan skala absolut/identitas kota.

**4.4 Target:** `Flood_t+{HORIZON}` = `groupby(city)["Flood"].shift(-HORIZON)`.

**4.5 Dropna:** buang baris yang `NaN` pada kolom lag/rolling/anomali/target. Cast target ke int.

### Tahap 4 — Definisi fitur final (eksplisit)
```
FEATURES_NUM =
  [ {var}_lag1..lag7  for var in [precip,tmax,tmin,wind,rain,soil] ]   # 42
  + [ rain_roll3_sum, rain_roll7_sum, rain_roll14_sum,
      precip_roll3_sum, precip_roll7_sum,
      soil_roll3_mean, soil_roll7_mean, tmax_roll7_mean ]               # 8
  + [ rain7_pctl, precip7_pctl, soil7_pctl ]                            # 3  (anomali)
FEATURES_CAT  = []                      # KOSONG — city TIDAK dipakai
ID_COLS       = [ city, time ]          # disimpan untuk split & filter, BUKAN fitur
TARGET        = Flood_t+{HORIZON}
```
Tulis hasil ke `processed/dataset_timeseries.csv` (kolom: ID_COLS + FEATURES_NUM + TARGET).

### Tahap 5 — Split time-aware (`global_date_cutoff`)
```
train = rows where time <  SPLIT_CUTOFF_DATE
test  = rows where time >= SPLIT_CUTOFF_DATE
```
Berlaku seragam untuk semua kota. Tidak ada baris training bertanggal ≥ cutoff. Reset index.

### Tahap 6 — Balancing (training only)
Pada **train** saja, dengan `SEED`:
1. Pisah `pos` (target=1) dan `neg` (target=0).
2. `easy_neg` = neg dengan `rain7_pctl < EASY_NEG_PERCENTILE/100`; `hard_neg` = sisanya.
3. Target jumlah neg = `UNDERSAMPLE_RATIO * len(pos)`.
4. **Pertahankan semua `hard_neg`**; kekurangan/kelebihan diatur dengan **men-sample `easy_neg`**
   (buang easy_neg lebih dulu). Jika `hard_neg` sudah > target → sample dari `hard_neg` (jarang).
5. `train_balanced = concat(pos, neg_terpilih)`, shuffle dengan `SEED`.
6. Set `class_weight="balanced"` / `scale_pos_weight` pada model.
> Test set TIDAK disentuh.

### Tahap 7 — Pemodelan
RandomForest, XGBoost, CatBoost + FLAML AutoML (`metric="ap"`/average_precision).
`StandardScaler` di-fit pada `train_balanced[FEATURES_NUM]` saja, transform train+test.

### Tahap 8 — Ablation (WAJIB) → `outputs/ablation_results.csv`
Latih dua skenario, **evaluasi pada test subset Kaltim yang sama** (`city ∈ KALTIM_CITIES`):

| Skenario | Training rows |
|---|---|
| `baseline` | `city ∈ KALTIM_CITIES` |
| `pooled`   | semua kota |

Metrik per skenario per model: `recall, precision, f1, pr_auc, balanced_acc` + confusion matrix.
Aturan keputusan: jika `pooled.recall(Kaltim) <= baseline.recall(Kaltim)` → laporkan jujur dan
lakukan ablation per-kota (keluarkan kota yang menurunkan metrik). Jangan sembunyikan hasil negatif.

### Tahap 9 — Modul Spasial (tidak berubah)
`MODE="geospatial"` tetap hanya `CITIES={Balikpapan,Samarinda}`. DEM/RBI **tidak** untuk kota luar.

---

## 5. Perubahan Konkret di `Flood_AutoML_Tabular.ipynb`

1. Sumber timeseries → `processed/dataset_timeseries.csv` (hasil Tahap 3-5).
2. Di `build_timeseries_table()`: ganti `cat_cols = [c for c in ["city"] if c in df.columns]`
   menjadi `cat_cols = []`. Pastikan `city` hanya di `id_cols`.
3. Tambah 3 fitur anomali (`rain7_pctl, precip7_pctl, soil7_pctl`) ke `num_cols`.
4. Split: gunakan `SPLIT_CUTOFF_DATE` (bukan rasio 0.70) untuk mode multi-kota.
5. Sisipkan fungsi undersample §4.6 setelah split, sebelum scaler.
6. Tambah blok ablation §4.8; simpan `outputs/ablation_results.csv`.
7. Saat evaluasi, filter `test_df[test_df.city.isin(KALTIM_CITIES)]` untuk metrik final.
8. Modul geospatial: tanpa perubahan.

---

## 6. Kriteria Penerimaan (Definition of Done)

- [ ] `dataset_utama_multikota.csv` berisi ≥ 6 kota, kolom sesuai §Tahap 1.6.
- [ ] `city_summary.csv` tercetak; tiap kota luar punya `n_flood > 0`.
- [ ] `dataset_timeseries.csv`: tidak ada `NaN` di FEATURES_NUM+TARGET; `FEATURES_CAT==[]`.
- [ ] Tidak ada kolom `city/lat/long` di dalam matriks fitur model.
- [ ] Split: `max(train.time) < SPLIT_CUTOFF_DATE <= min(test.time)`.
- [ ] Statistik scaler/persentil tidak memakai baris test (cek: persentil = expanding).
- [ ] `ablation_results.csv` berisi baris `baseline` & `pooled`, metrik dihitung pada subset Kaltim.
- [ ] Modul geospatial tetap hanya 2 kota; DEM/RBI kota luar tidak diakses.

---

## 7. Checklist Anti-Kebocoran

- [ ] Lag & rolling pakai `shift(1)` per `city`.
- [ ] Anomali = expanding (hanya masa lalu).
- [ ] Split berdasarkan `SPLIT_CUTOFF_DATE`, bukan random.
- [ ] Undersample/SMOTE/class-weight hanya di training.
- [ ] Test set tidak dibalance; metrik final hanya subset Kaltim.
- [ ] `city` bukan fitur; tidak ada proxy identitas kota di fitur.

---

## 8. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Model belajar "iklim kota" bukan pola banjir | Anomali per-kota; `city` di-drop; ablation |
| Banjir kota luar dipicu hulu, bukan cuaca lokal | Ablation per-kota; keluarkan kota penambah noise |
| Negatif eksklusif Kaltim → confound | Negatif diambil dari semua kota |
| Balance 1:1 → precision jeblok | `UNDERSAMPLE_RATIO=3` + class weight |
| Skema cuaca kota luar beda | Validasi §2.1; kota gagal di-skip + warning |
| Penguji: "kenapa data Bandung?" | Transfer pola fisik cuaca→banjir; kecamatan dari modul spasial Kaltim |

---

## 9. Default yang Dapat Ditimpa Pengguna

Jika pengguna menyebut nilai lain, timpa default berikut lalu lanjut eksekusi tanpa bertanya ulang:
`HORIZON`, `SPLIT_CUTOFF_DATE`, `UNDERSAMPLE_RATIO`, `EASY_NEG_PERCENTILE`, `NORMALIZATION`,
`EXTERNAL_CITIES`. Selain itu, ikuti dokumen ini apa adanya.
