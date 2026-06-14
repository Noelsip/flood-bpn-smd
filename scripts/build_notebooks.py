"""
build_notebooks.py
==================
Generator notebook tunggal: Flood_AutoML_Tabular.ipynb (di root repo).

Notebook OTOMATIS memilih jalur sesuai data yang tersedia:
  - DEM + RBI TIDAK ada (default repo ini)  -> MODE "timeseries":
    prediksi banjir 3 hari ke depan (Flood_t+3) dari dataset cuaca + banjir,
    lag feature t-1..t-7 + rolling, split berbasis waktu, AutoML.
  - DEM + RBI ADA (extract zip)             -> MODE "geospatial":
    titik terrain-constrained -> peta probabilitas banjir per kota.

Siap di-clone & dijalankan di Google Colab (ada sel pemasangan dependensi).

Jalankan ulang generator:  python scripts/build_notebooks.py
"""
import os
import nbformat as nbf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def guard(mode, body):
    """Bungkus body agar hanya jalan pada MODE tertentu (indentasi 4 spasi)."""
    inner = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.split("\n"))
    return f'if MODE == "{mode}":\n{inner}'


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    cells = []
    def md(t): cells.append(nbf.v4.new_markdown_cell(t))
    def code(t): cells.append(nbf.v4.new_code_cell(t))

    # ---------------------------------------------------------------- intro
    md("""# Flood AutoML — Balikpapan & Samarinda

Prediksi **kejadian banjir 3 hari ke depan** (`Flood_t+3`) dari data **cuaca + banjir**
yang ditransformasi menjadi **time series** (lag feature t-1..t-7 + akumulasi/rolling).

Notebook memilih jalur sendiri (`MODE`):

| Data tersedia | `MODE` | Yang dijalankan |
|---|---|---|
| **DEM + RBI tidak ada** (default) | `timeseries` | Time Series Classification banjir `t+3` dari cuaca + banjir. |
| **DEM + RBI ada** (extract zip ke `dem/`, `RBI/`, `padat-penduduk/`) | `geospatial` | Peta probabilitas banjir per kota dari DEM/RBI. |

**Cara pakai di Google Colab:**
```python
!git clone <URL-REPO-GITHUB-MU>.git
%cd machine-learning
```
lalu jalankan sel dari atas ke bawah.

Sifat penelitian (untuk sidang): **data time series**, **target klasifikasi biner** (0/1)
→ *Time Series Classification / Flood Forecasting Classification*, bukan forecasting nilai kontinu.""")

    # ---------------------------------------------------------------- install
    md("""## 0. Pasang dependensi
Otomatis hanya saat di Google Colab / environment baru. Di mesin yang sudah lengkap,
sel ini tidak melakukan apa-apa.""")
    code("""import sys, subprocess

def _in_colab():
    return "google.colab" in sys.modules

if _in_colab():
    pkgs = ["numpy", "pandas", "scikit-learn", "xgboost", "catboost",
            "flaml[automl]", "imbalanced-learn", "matplotlib"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    print("Dependensi terpasang (Colab).")
else:
    print("Bukan Colab: lewati pemasangan. Pastikan requirements.txt sudah terpasang.")""")

    # ---------------------------------------------------------------- imports
    md("## 1. Import")
    code("""import os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from flaml import AutoML

warnings.filterwarnings("ignore")

# SMOTE opsional. Bila tidak ada, otomatis fallback ke class weight.
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

SEED = 42
np.random.seed(SEED)
print("Import OK | SMOTE:", HAS_SMOTE)""")

    # ---------------------------------------------------------------- config + mode
    md("""## 2. Konfigurasi & deteksi MODE
Root proyek dicari otomatis (tahan dijalankan dari folder mana pun). `MODE = geospatial`
hanya bila DEM + RBI + admin + penduduk lengkap; selain itu `timeseries`.""")
    code("""def find_root(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / "processed" / "dataset_utama.csv").exists():
            return c
    for c in start.glob("*/processed/dataset_utama.csv"):
        return c.parent.parent
    return start

BASE_DIR = find_root(Path.cwd())
CLEAN_DIR = BASE_DIR / "clean"
ADMIN_DIR = CLEAN_DIR / "admin"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Root proyek:", BASE_DIR)

# --- parameter time series ---
HORIZON = 3                       # prediksi banjir 3 hari ke depan -> Flood_t+3
LAGS = 7                          # jendela observasi t-1..t-7
TARGET = f"Flood_t+{HORIZON}"
TS_PATH = BASE_DIR / "processed" / "dataset_timeseries.csv"
TS_BUILDER = BASE_DIR / "processed" / "build_timeseries_dataset.py"

# --- parameter umum / geospasial ---
TIME_BUDGET = 120                 # detik untuk AutoML FLAML
UTM_CRS, WGS84 = "EPSG:32750", "EPSG:4326"
DEM_PATH = BASE_DIR / "dem" / "DEM SRTM 30M KALIMANTAN TIMUR.tif"
CITIES = {
    "Balikpapan": dict(
        gdb=BASE_DIR / "RBI" / "KotaBalikpapan" / "RBI50K_KOTA BALIKPAPAN_KUGI50.gdb",
        banjir=CLEAN_DIR / "banjir_balikpapan.csv", penduduk=CLEAN_DIR / "penduduk_balikpapan.csv",
        admin_kec=ADMIN_DIR / "balikpapan_kecamatan.geojson", admin_kel=ADMIN_DIR / "balikpapan_kelurahan.geojson"),
    "Samarinda": dict(
        gdb=BASE_DIR / "RBI" / "KotaSamarinda" / "RBI50K_KOTA SAMARINDA_KUGI50.gdb",
        banjir=CLEAN_DIR / "banjir_samarinda.csv", penduduk=CLEAN_DIR / "penduduk_samarinda.csv",
        admin_kec=ADMIN_DIR / "samarinda_kecamatan.geojson", admin_kel=ADMIN_DIR / "samarinda_kelurahan.geojson"),
}
ADMIN_KEC_NAMECOL, ADMIN_KEL_NAMECOL = "WADMKC", "WADMKD"
L_RIVER, L_ROAD, L_LANDCOVER, LC_CLASS_COL = "SUNGAI_LN_50K", "JALAN_LN_50K", "PENUTUPLAHAN_AR_50K", "REMARK"
POINTS_PER_EVENT, FLOOD_ELEV_QUANTILE, NONFLOOD_ELEV_QUANTILE, NONFLOOD_MIN_DIST = 8, 0.40, 0.50, 750.0
NUMERIC_COLS, CAT_COLS = ["elevation", "slope", "dist_river", "dist_road", "pop_density"], ["landcover"]
FEATURE_COLS = NUMERIC_COLS + CAT_COLS

def _geo_ready():
    if not DEM_PATH.exists():
        return False
    for cfg in CITIES.values():
        for key in ("gdb", "admin_kec", "admin_kel", "banjir", "penduduk"):
            if not Path(cfg[key]).exists():
                return False
    return True

MODE = "geospatial" if _geo_ready() else "timeseries"
print("MODE:", MODE)
if MODE == "timeseries":
    print("-> DEM/RBI tidak ada: prediksi dari dataset cuaca + banjir (Flood_t+3).")
else:
    print("-> Data geospasial lengkap: membangun peta probabilitas banjir.")""")

    # ================================================================
    # JALUR A — TIME SERIES (cuaca + banjir)
    # ================================================================
    md("""---
# A. Jalur Time Series — cuaca + banjir
Sel di bawah hanya jalan saat `MODE == "timeseries"`.""")

    md("""### A.1 Muat dataset time series
`dataset_timeseries.csv` sudah disertakan di repo. Bila belum ada, dibangun otomatis dari
`dataset_utama.csv` (lag t-1..t-7 + rolling + target Flood_t+3).""")
    code(guard("timeseries", """if not TS_PATH.exists():
    print("Membangun dataset time series...")
    subprocess.run([sys.executable, str(TS_BUILDER)], check=True)
df = pd.read_csv(TS_PATH, parse_dates=["time"]).sort_values(["time", "city"]).reset_index(drop=True)
print("Dataset:", df.shape, "| target:", TARGET,
      "|", df["time"].min().date(), "->", df["time"].max().date())
display(df.head(3))"""))

    md("### A.2 Eksplorasi: kelas, wilayah, waktu")
    code(guard("timeseries", """fig, ax = plt.subplots(1, 2, figsize=(13, 4))
vc = df[TARGET].value_counts().sort_index()
vc.rename({0: "Tidak Banjir", 1: "Banjir"}).plot(kind="bar", ax=ax[0], color=["#4C9F70", "#D1495B"])
ax[0].set_title(f"Keseimbangan kelas ({TARGET})"); ax[0].tick_params(axis="x", rotation=0)
for i, v in enumerate(vc.values):
    ax[0].text(i, v, str(v), ha="center", va="bottom")
df[df[TARGET] == 1].assign(year=df["time"].dt.year).groupby("year").size().plot(
    kind="bar", ax=ax[1], color="#D1495B")
ax[1].set_title("Hari berlabel banjir (t+3) per tahun"); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()
pos = int(df[TARGET].sum())
print(f"Banjir: {pos}/{len(df)} ({pos/len(df)*100:.2f}%) -> imbalance berat.")

region_cols = [c for c in ["kec_flood_total", "kec_flood_max", "kec_affected", "kec_total"] if c in df.columns]
if region_cols:
    print("\\nFitur wilayah per kota (banjir per kecamatan, konteks statis):")
    display(df.groupby("city")[region_cols].first())
for nm, fn in [("Balikpapan", "clean/banjir_balikpapan.csv"), ("Samarinda", "clean/banjir_samarinda.csv")]:
    p = BASE_DIR / fn
    if p.exists():
        print(f"\\n== Banjir per kecamatan — {nm} ==")
        print(pd.read_csv(p).sort_values("banjir_count", ascending=False).to_string(index=False))"""))

    md("""### A.3 Fitur, target, split berbasis waktu, preprocessing
Fitur = lag (t-1..t-7) + rolling + kalender + wilayah; `city` sebagai kategori.
Split 70% awal / 30% terbaru (anti-leakage). Scaler & encoder di-fit **hanya di train**.""")
    code(guard("timeseries", """drop_cols = {"location_id", "latitude", "longitude", "time", "Flood", TARGET}
cat_cols = [c for c in ["city"] if c in df.columns]
num_cols = [c for c in df.columns if c not in drop_cols and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
feature_cols = num_cols + cat_cols
print(f"Fitur: {len(feature_cols)} ({len(num_cols)} numerik + {len(cat_cols)} kategori)")

ordered = df.sort_values("time").reset_index(drop=True)
cut = int(len(ordered) * 0.70)
train_df, test_df = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
print("Train:", train_df["time"].min().date(), "->", train_df["time"].max().date(),
      "|", len(train_df), "baris,", int(train_df[TARGET].sum()), "banjir")
print("Test :", test_df["time"].min().date(), "->", test_df["time"].max().date(),
      "|", len(test_df), "baris,", int(test_df[TARGET].sum()), "banjir")

scaler = StandardScaler().fit(train_df[num_cols])
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train_df[cat_cols]) if cat_cols else None
def transform(frame):
    parts = [pd.DataFrame(scaler.transform(frame[num_cols]), columns=num_cols, index=frame.index)]
    if encoder is not None:
        parts.append(pd.DataFrame(encoder.transform(frame[cat_cols]), columns=cat_cols, index=frame.index))
    return pd.concat(parts, axis=1)
X_train = transform(train_df).reset_index(drop=True)
X_test = transform(test_df).reset_index(drop=True)
y_train = train_df[TARGET].astype(int).reset_index(drop=True)
y_test = test_df[TARGET].astype(int).reset_index(drop=True)
print("X_train:", X_train.shape, "| X_test:", X_test.shape)"""))

    md("""### A.4 Penanganan imbalance — hanya pada data latih
SMOTE bila tersedia (oversample banjir di train), jika tidak fallback ke class weight.
Data uji tidak disentuh.""")
    code(guard("timeseries", """neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
scale_pos_weight = neg / max(pos, 1)
if HAS_SMOTE and pos >= 6:
    k = min(5, pos - 1)
    X_fit, y_fit = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_train, y_train)
    imbalance_method, spw = f"SMOTE (k={k})", 1.0
else:
    X_fit, y_fit, imbalance_method, spw = X_train, y_train, "class_weight", scale_pos_weight
print("Imbalance:", imbalance_method, "| sebaran y_fit:",
      dict(pd.Series(y_fit).value_counts().sort_index()), "| spw:", round(spw, 2))"""))

    md("### A.5 Model baseline + AutoML (dioptimasi ke AUC-PR)")
    code(guard("timeseries", """models = {}
models["Random Forest"] = RandomForestClassifier(
    n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample").fit(X_fit, y_fit)
models["XGBoost"] = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", scale_pos_weight=spw, random_state=SEED, n_jobs=-1).fit(X_fit, y_fit)
models["CatBoost"] = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05, random_seed=SEED, verbose=0,
    allow_writing_files=False, class_weights=[1.0, spw]).fit(X_fit, y_fit)

automl = AutoML()
for _metric, _ests in [("ap", ["rf", "xgboost", "catboost"]),
                       ("roc_auc", ["rf", "xgboost", "catboost"]),
                       ("roc_auc", ["rf", "xgboost"])]:
    try:
        automl.fit(X_train=X_fit, y_train=y_fit, task="classification", metric=_metric,
                   estimator_list=_ests, time_budget=TIME_BUDGET, eval_method="cv",
                   n_splits=5, seed=SEED, verbose=1)
        break
    except Exception as e:
        print(f"AutoML gagal (metric={_metric}, est={_ests}): {e}")
models["AutoML (FLAML)"] = automl
print("Estimator terbaik:", automl.best_estimator)"""))

    md("""### A.6 Evaluasi (fokus Recall & AUC-PR)
Untuk data timpang, Recall & AUC-PR lebih informatif dari accuracy. Model terbaik dipilih dari AUC-PR.""")
    code(guard("timeseries", """def proba(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]
rows, results = {}, []
for name, m in models.items():
    p = proba(m, X_test); rows[name] = p; pred = (p >= 0.5).astype(int)
    results.append(dict(Model=name, Accuracy=accuracy_score(y_test, pred),
                        Balanced_Acc=balanced_accuracy_score(y_test, pred),
                        Precision=precision_score(y_test, pred, zero_division=0),
                        Recall=recall_score(y_test, pred, zero_division=0),
                        F1=f1_score(y_test, pred, zero_division=0),
                        AUC_ROC=roc_auc_score(y_test, p) if y_test.nunique() > 1 else np.nan,
                        AUC_PR=average_precision_score(y_test, p) if y_test.nunique() > 1 else np.nan))
res_df = pd.DataFrame(results).set_index("Model").sort_values("AUC_PR", ascending=False)
best_name = res_df.index[0]; best_model = models[best_name]
print("Model terbaik (AUC-PR):", best_name); display(res_df.round(4))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name, p in rows.items():
    if y_test.nunique() > 1:
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, lw=2, label=f"{name} ({average_precision_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
ax[1].set_title("Precision-Recall"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()
plt.tight_layout(); plt.show()

best_pred = (proba(best_model, X_test) >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_pred)
fig, axc = plt.subplots(figsize=(4.5, 4)); axc.imshow(cm, cmap="Blues")
axc.set_xticks([0, 1]); axc.set_xticklabels(["Tidak Banjir", "Banjir"])
axc.set_yticks([0, 1]); axc.set_yticklabels(["Tidak Banjir", "Banjir"])
axc.set_xlabel("Prediksi"); axc.set_ylabel("Aktual"); axc.set_title(f"Confusion Matrix — {best_name}")
for (r, c), v in np.ndenumerate(cm):
    axc.text(c, r, str(v), ha="center", va="center", color="white" if v > cm.max()/2 else "black")
plt.tight_layout(); plt.show()
print(classification_report(y_test, best_pred, target_names=["Tidak Banjir", "Banjir"], zero_division=0))"""))

    md("### A.7 Feature importance & simpan hasil")
    code(guard("timeseries", """frames = []
for name, m in models.items():
    imp = getattr(m, "feature_importances_", None)
    if imp is None and name == "AutoML (FLAML)":
        est = getattr(getattr(m, "model", None), "estimator", None)
        imp = getattr(est, "feature_importances_", None)
    if imp is not None and len(imp) == X_train.shape[1]:
        frames.append(pd.Series(np.asarray(imp) / (np.sum(imp) + 1e-12), index=X_train.columns, name=name))
if frames:
    imp_df = pd.concat(frames, axis=1); imp_df["mean"] = imp_df.mean(axis=1)
    imp_df.sort_values("mean").tail(20)["mean"].plot(kind="barh", figsize=(8, 7), color="#2A6F97")
    plt.title("20 fitur paling berpengaruh"); plt.xlabel("Importance"); plt.tight_layout(); plt.show()
    display(imp_df.sort_values("mean", ascending=False).head(15).round(4))

res_df.to_csv(OUT_DIR / "timeseries_model_comparison.csv")
ordered[feature_cols + [TARGET, "time"]].to_csv(OUT_DIR / "timeseries_training_table.csv", index=False)
print("Tersimpan di:", OUT_DIR)"""))

    # ================================================================
    # JALUR B — GEOSPASIAL (DEM + RBI)
    # ================================================================
    md("""---
# B. Jalur Geospasial — DEM + RBI
Hanya jalan bila `MODE == "geospatial"` (DEM/RBI lengkap). Untuk repo ringan ini dilewati otomatis.""")

    md("### B.1 Import & fungsi geospasial")
    code(guard("geospatial", '''import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, box

def assert_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Tidak ditemukan: {path}")
def _norm(s):
    return str(s).strip().lower()
def read_gdb(gdb, layer):
    assert_file(gdb); g = gpd.read_file(gdb, layer=layer)
    return (g.set_crs(WGS84) if g.crs is None else g).to_crs(UTM_CRS)
def load_admin(path, name_col):
    assert_file(path); g = gpd.read_file(path)
    if name_col not in g.columns:
        raise KeyError(f"Kolom {name_col!r} tidak ada di {Path(path).name}. Kolom: {list(g.columns)}")
    g = (g.set_crs(WGS84) if g.crs is None else g).to_crs(UTM_CRS).copy(); g["adm_name"] = g[name_col]; return g
def city_boundary_from_admin(kec):
    return kec.dissolve().geometry.iloc[0]
def clip_to_city(gdf, boundary):
    return gdf[gdf.intersects(boundary)].copy()
def clip_dem_to_bounds(dem_path, boundary_wgs, dst_crs=UTM_CRS, pad=0.02):
    minx, miny, maxx, maxy = boundary_wgs.bounds; bb = box(minx-pad, miny-pad, maxx+pad, maxy+pad)
    with rasterio.open(dem_path) as src:
        arr, tr = rio_mask(src, [bb], crop=True)
        t2, w2, h2 = calculate_default_transform(src.crs, dst_crs, arr.shape[2], arr.shape[1],
            *rasterio.transform.array_bounds(arr.shape[1], arr.shape[2], tr))
        dst = np.full((h2, w2), np.nan, dtype="float32")
        reproject(source=arr[0], destination=dst, src_transform=tr, src_crs=src.crs,
                  dst_transform=t2, dst_crs=dst_crs, resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return dst.astype(float), t2
def compute_slope_deg(dem, transform):
    px, py = abs(transform.a), abs(transform.e); dzdy, dzdx = np.gradient(dem, py, px)
    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
def sample_array_at_points(points, array, transform):
    vals = []
    for geom in points.geometry:
        r, c = rowcol(transform, geom.x, geom.y)
        vals.append(array[r, c] if (0 <= r < array.shape[0] and 0 <= c < array.shape[1]) else np.nan)
    return np.array(vals, dtype=float)
def attach_polygon_value(points, poly_gdf, value_col):
    j = gpd.sjoin(points, poly_gdf[["geometry", value_col]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]; return j.loc[points.index, value_col].values
def sample_flood_points_in_kecamatan(poly, n, dem, transform, elev_q, rng):
    H, W = dem.shape
    mask = rasterize([(poly, 1)], out_shape=(H, W), transform=transform, fill=0, all_touched=True).astype(bool)
    cells = mask & np.isfinite(dem)
    if cells.sum() == 0:
        return []
    thr = np.nanquantile(dem[cells], elev_q); low = cells & (dem <= thr)
    rows, cols = np.where(low if low.sum() >= n else cells)
    if len(rows) == 0:
        return []
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=len(rows) < n)
    return [Point(*rasterio.transform.xy(transform, rows[k], cols[k])) for k in idx]
def generate_nonflood_points(flood_gdf, dem, transform, n, elev_q, min_dist, rng):
    minx, miny, maxx, maxy = flood_gdf.total_bounds; pad = 0.15 * max(maxx-minx, maxy-miny)
    minx, miny, maxx, maxy = minx-pad, miny-pad, maxx+pad, maxy+pad
    thr = np.nanquantile(dem, elev_q); flood_union = flood_gdf.unary_union; pts, tries = [], 0
    while len(pts) < n and tries < n * 400:
        tries += 1; px, py = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        r, c = rowcol(transform, px, py)
        if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
            continue
        if not np.isfinite(dem[r, c]) or dem[r, c] < thr:
            continue
        p = Point(px, py)
        if p.distance(flood_union) < min_dist:
            continue
        pts.append(p)
    return gpd.GeoDataFrame(geometry=pts, crs=UTM_CRS)
print("Fungsi geospasial siap.")'''))

    md("### B.2 Bangun dataset berlabel per kota")
    code(guard("geospatial", '''CITY_LAYERS, RAW_DFS = {}, []
def build_city_dataset(name, cfg):
    print(f"\\n========== {name} =========="); rng = np.random.default_rng(SEED)
    kec = load_admin(cfg["admin_kec"], ADMIN_KEC_NAMECOL); desakel = load_admin(cfg["admin_kel"], ADMIN_KEL_NAMECOL)
    bnd_utm = city_boundary_from_admin(kec); bnd_wgs = gpd.GeoSeries([bnd_utm], crs=UTM_CRS).to_crs(WGS84).iloc[0]
    dem, tr = clip_dem_to_bounds(DEM_PATH, bnd_wgs); slope = compute_slope_deg(dem, tr)
    rivers = clip_to_city(read_gdb(cfg["gdb"], L_RIVER), bnd_utm)
    roads = clip_to_city(read_gdb(cfg["gdb"], L_ROAD), bnd_utm)
    landcover = clip_to_city(read_gdb(cfg["gdb"], L_LANDCOVER), bnd_utm)
    pen = pd.read_csv(cfg["penduduk"]); pen["_key"] = pen["kelurahan"].map(_norm)
    desakel["_key"] = desakel["adm_name"].map(_norm); desakel["area_km2"] = desakel.geometry.area / 1e6
    desakel = desakel.merge(pen[["_key", "value", "value_type"]].drop_duplicates("_key"), on="_key", how="left")
    desakel["pop_density"] = np.where(desakel["value_type"].eq("count"), desakel["value"]/desakel["area_km2"], desakel["value"])
    kec["_key"] = kec["adm_name"].map(_norm); banjir = pd.read_csv(cfg["banjir"]); banjir["_key"] = banjir["kecamatan"].map(_norm)
    kec = kec.merge(banjir[["_key", "banjir_count"]], on="_key", how="left")
    flood_pts = []
    for _, row in kec.iterrows():
        cnt = row["banjir_count"]
        if pd.isna(cnt) or cnt <= 0:
            continue
        flood_pts += sample_flood_points_in_kecamatan(row.geometry, int(round(cnt))*POINTS_PER_EVENT, dem, tr, FLOOD_ELEV_QUANTILE, rng)
    flood_gdf = gpd.GeoDataFrame(geometry=flood_pts, crs=UTM_CRS)
    nonflood_gdf = generate_nonflood_points(flood_gdf, dem, tr, len(flood_gdf), NONFLOOD_ELEV_QUANTILE, NONFLOOD_MIN_DIST, rng)
    flood_gdf["label"] = 1; nonflood_gdf["label"] = 0
    pts = gpd.GeoDataFrame(pd.concat([flood_gdf[["geometry", "label"]], nonflood_gdf[["geometry", "label"]]], ignore_index=True),
                           geometry="geometry", crs=UTM_CRS)
    rivers_u, roads_u = rivers.unary_union, roads.unary_union
    pts["elevation"] = sample_array_at_points(pts, dem, tr); pts["slope"] = sample_array_at_points(pts, slope, tr)
    pts["dist_river"] = pts.geometry.apply(lambda g: g.distance(rivers_u))
    pts["dist_road"] = pts.geometry.apply(lambda g: g.distance(roads_u))
    pts["landcover_raw"] = [str(v) for v in attach_polygon_value(pts, landcover, LC_CLASS_COL)]
    pts["pop_density"] = attach_polygon_value(pts, desakel, "pop_density"); pts["city"] = name
    out = pd.DataFrame(pts.drop(columns="geometry")); out["x"], out["y"] = pts.geometry.x.values, pts.geometry.y.values
    print(f"  banjir={int((out['label']==1).sum())} non-banjir={int((out['label']==0).sum())}")
    return out, dict(dem=dem, slope=slope, transform=tr, rivers=rivers, roads=roads, landcover=landcover, desakel=desakel, boundary=bnd_utm)
for nm, cfg in CITIES.items():
    d, lyr = build_city_dataset(nm, cfg); RAW_DFS.append(d); CITY_LAYERS[nm] = lyr
df_all = pd.concat(RAW_DFS, ignore_index=True)
lc_classes = sorted(df_all["landcover_raw"].dropna().unique().tolist()); LC_MAP = {c: i for i, c in enumerate(lc_classes)}
df_all["landcover"] = df_all["landcover_raw"].map(LC_MAP).fillna(-1).astype(int)
print("\\nDataset gabungan:", df_all.shape, "| kelas:", df_all["label"].value_counts().to_dict()); display(df_all.head())'''))

    md("### B.3 Preprocessing, model, AutoML, evaluasi")
    code(guard("geospatial", '''df = df_all.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
scaler = StandardScaler().fit(df[NUMERIC_COLS])
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(df[CAT_COLS])
def transform_features(frame):
    Xn = pd.DataFrame(scaler.transform(frame[NUMERIC_COLS]), columns=NUMERIC_COLS, index=frame.index)
    Xc = pd.DataFrame(encoder.transform(frame[CAT_COLS]), columns=CAT_COLS, index=frame.index)
    return pd.concat([Xn, Xc], axis=1)
X = transform_features(df); y = df["label"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)

models = {}
models["Random Forest"] = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
models["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1).fit(X_train, y_train)
models["CatBoost"] = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, random_seed=SEED,
    verbose=0, allow_writing_files=False).fit(X_train, y_train)
automl = AutoML()
for _ests in [["rf", "xgboost", "catboost"], ["rf", "xgboost"]]:
    try:
        automl.fit(X_train=X_train, y_train=y_train, task="classification", metric="roc_auc",
                   estimator_list=_ests, time_budget=TIME_BUDGET, eval_method="cv", n_splits=5, seed=SEED, verbose=1)
        break
    except Exception as e:
        print("AutoML gagal:", e)
models["AutoML (FLAML)"] = automl

def get_proba(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]
results, probas = [], {}
for name, m in models.items():
    p = get_proba(m, X_test); probas[name] = p; pred = (p >= 0.5).astype(int)
    results.append(dict(Model=name, Accuracy=accuracy_score(y_test, pred),
        Precision=precision_score(y_test, pred, zero_division=0), Recall=recall_score(y_test, pred, zero_division=0),
        F1=f1_score(y_test, pred, zero_division=0), AUC_ROC=roc_auc_score(y_test, p)))
res_df = pd.DataFrame(results).set_index("Model").sort_values("AUC_ROC", ascending=False)
best_name = res_df.index[0]; best_model = models[best_name]
print("Model terbaik (AUC-ROC):", best_name); display(res_df.round(4))
print(classification_report(y_test, (get_proba(best_model, X_test) >= 0.5).astype(int), target_names=["Non-Banjir", "Banjir"]))'''))

    md("### B.4 Peta probabilitas banjir per kota (GeoTIFF + PNG)")
    code(guard("geospatial", '''def build_probability_map(name, lyr):
    dem, slope, tr = lyr["dem"], lyr["slope"], lyr["transform"]
    rivers, roads, landcover, desakel = lyr["rivers"], lyr["roads"], lyr["landcover"], lyr["desakel"]
    H, W = dem.shape; px = abs(tr.a)
    river_mask = rasterize(((g, 1) for g in rivers.geometry), out_shape=(H, W), transform=tr, fill=0, all_touched=True)
    road_mask = rasterize(((g, 1) for g in roads.geometry), out_shape=(H, W), transform=tr, fill=0, all_touched=True)
    dist_river_grid = distance_transform_edt(river_mask == 0) * px; dist_road_grid = distance_transform_edt(road_mask == 0) * px
    lc_grid = rasterize(((g, LC_MAP.get(str(v), -1)) for g, v in zip(landcover.geometry, landcover[LC_CLASS_COL])),
                        out_shape=(H, W), transform=tr, fill=-1, dtype="float32")
    pop_grid = rasterize(((g, float(v)) for g, v in zip(desakel.geometry, desakel["pop_density"]) if pd.notna(v)),
                         out_shape=(H, W), transform=tr, fill=np.nan, dtype="float32")
    grid = pd.DataFrame({"elevation": dem.ravel(), "slope": slope.ravel(), "dist_river": dist_river_grid.ravel(),
                         "dist_road": dist_road_grid.ravel(), "pop_density": pop_grid.ravel(), "landcover": lc_grid.ravel().astype(int)})
    valid = grid[NUMERIC_COLS].notna().all(axis=1).values; prob = np.full(len(grid), np.nan)
    if valid.sum() > 0:
        prob[valid] = get_proba(best_model, transform_features(grid[valid]))
    prob_map = prob.reshape(H, W); tif = OUT_DIR / f"flood_probability_{name}.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
                       crs=UTM_CRS, transform=tr, nodata=np.nan) as dst:
        dst.write(prob_map.astype("float32"), 1)
    plt.figure(figsize=(8, 8)); im = plt.imshow(prob_map, cmap="turbo", vmin=0, vmax=1)
    plt.colorbar(im, label="Probabilitas banjir"); plt.title(f"Peta Probabilitas — {name} ({best_name})"); plt.axis("off")
    png = OUT_DIR / f"flood_probability_{name}.png"; plt.savefig(png, dpi=150, bbox_inches="tight"); plt.show()
    print("Tersimpan:", tif.name, "&", png.name)
for nm, lyr in CITY_LAYERS.items():
    build_probability_map(nm, lyr)
res_df.to_csv(OUT_DIR / "geospatial_model_comparison.csv"); df.to_csv(OUT_DIR / "geospatial_labelled_points.csv", index=False)
print("Output di:", OUT_DIR)'''))

    md("""---
## Catatan metodologi (untuk laporan/sidang)
**Time series (cuaca + banjir):**
- Data harian → **lag feature** t-1..t-7 + **akumulasi** (hujan 3/7/14 hari, dll). Target = banjir **t+3**.
- **Klasifikasi biner** atas data **time series** (Time Series Classification), bukan forecasting kontinu.
- Split **berbasis waktu**; **imbalance** ditangani hanya di data latih (SMOTE/class weight) **setelah** split.
- **Wilayah** (banjir per kecamatan BPS) diikutkan sebagai konteks per kota (tahun + jumlah, tanpa time series).
- Metrik utama: **Recall** & **AUC-PR**.

**Geospasial (DEM + RBI):**
- 6 conditioning factor: elevation, slope, dist_river, dist_road, landcover, pop_density.
- Label BPS level **kecamatan**; titik banjir *terrain-constrained* (elevasi rendah ∝ intensitas).
- DEM dipotong per kota lalu direproyeksi UTM 50S (slope & jarak dalam meter).""")

    nb["cells"] = cells
    return nb


def main():
    nb = build_notebook()
    out_path = os.path.join(BASE, "Flood_AutoML_Tabular.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook ditulis:", out_path, "|", len(nb["cells"]), "sel")


if __name__ == "__main__":
    main()
