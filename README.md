# Flood Probability Estimation (AutoML) — Balikpapan & Samarinda

Implementasi lokal (Windows) dari paper *Flood Probability Estimation using AutoML in
Balikpapan and Samarinda*. Memakai data asli: DEM SRTM, RBI (sungai/jalan/land cover),
data banjir BPS, dan kepadatan penduduk.

## Struktur

```
machine-learning/
├─ Flood_AutoML_Local.ipynb     <- NOTEBOOK UTAMA (jalankan ini)
├─ requirements.txt
├─ scripts/
│  ├─ 01_clean_tabular.py        <- bersihkan banjir + penduduk -> clean/*.csv
│  └─ 02_build_notebook.py       <- generator notebook (sudah dijalankan)
├─ clean/                        <- HASIL pembersihan (sudah dibuat)
│  ├─ banjir_balikpapan.csv      banjir_samarinda.csv
│  ├─ penduduk_balikpapan.csv    penduduk_samarinda.csv
│  └─ admin/                     <- ⚠️ KAMU ISI: batas kecamatan & kelurahan kota
├─ dem/   RBI/   banjir/   padat-penduduk/   <- data mentah
└─ outputs/                      <- hasil model & peta (dibuat saat notebook jalan)
```

## Cara menjalankan

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts\01_clean_tabular.py     # sudah diuji, menghasilkan folder clean/
jupyter notebook Flood_AutoML_Local.ipynb   # jalankan sel atas->bawah
```

## ⚠️ Yang HARUS kamu lengkapi sebelum notebook bisa jalan penuh

### 1. Batas administrasi kota (WAJIB) — penyebab utama
Layer administrasi di RBI `.gdb` bawaan **TIDAK memuat Kota Balikpapan/Samarinda**
(hanya kabupaten tetangga: Kutai Kartanegara, Penajam Paser Utara). Layer spasial lain
(sungai, jalan, land cover) **sudah benar** menutupi kota. Jadi yang kurang **hanya batas admin**.

Sediakan 4 file di `clean/admin/` (format .geojson / .shp / .gpkg):
- `balikpapan_kecamatan.geojson`, `balikpapan_kelurahan.geojson`
- `samarinda_kecamatan.geojson`,  `samarinda_kelurahan.geojson`

Sumber:
- **BIG / tanahair.indonesia.go.id** → "Batas Wilayah Administrasi Desa" Kota Balikpapan & Samarinda (paling resmi).
- **GADM** (gadm.org) Indonesia level 3 (kecamatan) & level 4 (kelurahan) → potong ke 2 kota.

Lalu di **Sel 1** notebook, set `ADMIN_KEC_NAMECOL` / `ADMIN_KEL_NAMECOL` ke nama kolom
sumbermu (RBI: `WADMKC`/`WADMKD`; GADM: `NAME_3`/`NAME_4`).

### 2. Penduduk Balikpapan (disarankan)
`clean/penduduk_balikpapan.csv` baru mencakup 3 dari 6 kecamatan
(Barat, Utara, Kota). Kecamatan **Selatan, Timur, Tengah** belum ada → `pop_density` NaN
untuk titik di sana. Lengkapi data penduduk per kelurahan-nya bila tersedia.

## Catatan metodologi (untuk laporan/skripsi)
- Model terbaik dipilih dari **AUC-ROC** (sesuai paper), + Precision/Recall/F1.
- **Keterbatasan label banjir:** data BPS hanya level **kecamatan** (jumlah kelurahan
  terdampak), bukan koordinat. Titik banjir ditempatkan *terrain-constrained* (zona
  elevasi rendah dalam kecamatan terdampak, jumlah ∝ intensitas). Dokumentasikan asumsi ini.
- 6 conditioning factor: elevation, slope, dist_river, dist_road, landcover, pop_density.
  Untuk publikasi, pertimbangkan menambah **curah hujan** & **TWI** (dari DEM).
```
