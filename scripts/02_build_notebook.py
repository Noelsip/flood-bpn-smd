"""
02_build_notebook.py
====================
Membangun notebook `Flood_AutoML_Local.ipynb` (versi LOKAL Windows) dari paper
Flood AutoML, memakai data asli di repo ini:
  - DEM SRTM  (dem/…tif)               -> elevation, slope
  - RBI .gdb  (RBI/Kota…/*.gdb)        -> sungai, jalan, land cover, batas kelurahan
  - clean/banjir_*.csv                 -> label banjir (level kecamatan)
  - clean/penduduk_*.csv               -> kepadatan penduduk (level kelurahan)

Jalankan generator ini SEKALI untuk menghasilkan notebook:
    python scripts/02_build_notebook.py
Lalu buka Flood_AutoML_Local.ipynb dan jalankan sel dari atas ke bawah.
"""
import os
import nbformat as nbf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()
cells = []
def md(t):   cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# ───────────────────────────────────────────────────────────────────────────
md("""# Flood Probability Estimation using AutoML — Balikpapan & Samarinda (versi LOKAL)

Implementasi paper *"Flood Probability Estimation using Automated Machine Learning (AutoML)
in Balikpapan and Samarinda"* memakai **data asli** di komputer lokal (Windows).

Perbedaan dari versi Colab:
- Tidak memakai Google Drive / `google.colab`.
- Membaca **RBI File Geodatabase (`.gdb`)** langsung (bukan `.shp`), difilter per kota.
- Label banjir dari **BPS** (level kecamatan, intensitas = jumlah kelurahan terdampak);
  titik banjir disebar di zona **elevasi rendah** dalam tiap kecamatan terdampak
  (*terrain-constrained inventory sampling*).
- Kepadatan penduduk dihitung di **level kelurahan**.

> **Prasyarat:** jalankan dulu `python scripts/01_clean_tabular.py` (menghasilkan folder `clean/`).
> Jalankan sel notebook ini berurutan dari atas ke bawah.""")

# 0. Setup ------------------------------------------------------------------
md("## 0. Setup environment")
code("""# Jalankan sekali bila library belum terpasang (lihat juga requirements.txt)
# %pip install geopandas rasterio pyogrio shapely pyproj rtree \\
#              flaml[automl] catboost xgboost shap scikit-learn matplotlib openpyxl
print("Lewati sel ini jika environment sudah siap.")""")

code("""import os, warnings
# --- Perbaiki konflik PROJ: buang PROJ_LIB/PROJ_DATA/GDAL_DATA dari instalasi lain
#     (mis. PostgreSQL/PostGIS) agar rasterio & pyproj pakai database bawaannya sendiri.
for _k in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_k, None)
import pyproj
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()   # arahkan ke proj.db pyproj yang valid
pyproj.datadir.set_data_dir(pyproj.datadir.get_data_dir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from flaml import AutoML

import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, box

try:
    import shap; HAS_SHAP = True
except Exception:
    HAS_SHAP = False
print("Import OK. SHAP:", HAS_SHAP)""")

# 3. Config -----------------------------------------------------------------
md("""## 1. Konfigurasi
Sesuaikan **BASE_DIR** bila notebook tidak berada di folder proyek. Semua path lain otomatis.""")
code('''# ============================ KONFIGURASI ============================
BASE_DIR = os.getcwd()                       # << folder proyek (tempat notebook ini)
CLEAN_DIR = os.path.join(BASE_DIR, "clean")  # output dari scripts/01_clean_tabular.py
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
TIME_BUDGET = 120                 # anggaran waktu AutoML FLAML (detik)
UTM_CRS = "EPSG:32750"            # WGS84 / UTM 50S (Kaltim) -> meter
WGS84   = "EPSG:4326"
np.random.seed(RANDOM_SEED)

DEM_PATH = os.path.join(BASE_DIR, "dem", "DEM SRTM 30M KALIMANTAN TIMUR.tif")

# Tiap kota: path .gdb + nama resmi di kolom WADMKK + file clean banjir & penduduk
# PENTING: layer administrasi DI DALAM .gdb RBI ini TIDAK memuat kota targetnya
# (hanya kabupaten tetangga). Maka batas kecamatan & kelurahan kota HARUS diambil
# dari file eksternal (.shp/.geojson/.gpkg) yang kamu sediakan di folder clean/admin/.
# Sumber: BIG / tanahair.indonesia.go.id (batas desa), atau GADM (gadm.org) L3/L4.
ADMIN_DIR = os.path.join(CLEAN_DIR, "admin")

CITIES = {
    "Balikpapan": dict(
        gdb=os.path.join(BASE_DIR, "RBI", "KotaBalikpapan", "RBI50K_KOTA BALIKPAPAN_KUGI50.gdb"),
        kabkota="Kota Balikpapan",
        banjir=os.path.join(CLEAN_DIR, "banjir_balikpapan.csv"),
        penduduk=os.path.join(CLEAN_DIR, "penduduk_balikpapan.csv"),
        admin_kec=os.path.join(ADMIN_DIR, "balikpapan_kecamatan.geojson"),  # << sediakan
        admin_kel=os.path.join(ADMIN_DIR, "balikpapan_kelurahan.geojson"),  # << sediakan
    ),
    "Samarinda": dict(
        gdb=os.path.join(BASE_DIR, "RBI", "KotaSamarinda", "RBI50K_KOTA SAMARINDA_KUGI50.gdb"),
        kabkota="Kota Samarinda",
        banjir=os.path.join(CLEAN_DIR, "banjir_samarinda.csv"),
        penduduk=os.path.join(CLEAN_DIR, "penduduk_samarinda.csv"),
        admin_kec=os.path.join(ADMIN_DIR, "samarinda_kecamatan.geojson"),   # << sediakan
        admin_kel=os.path.join(ADMIN_DIR, "samarinda_kelurahan.geojson"),   # << sediakan
    ),
}

# Nama kolom NAMA wilayah di file admin eksternal. Sesuaikan ke sumbermu:
#   RBI/BIG -> kecamatan "WADMKC", kelurahan "WADMKD"
#   GADM    -> kecamatan "NAME_3", kelurahan "NAME_4"
ADMIN_KEC_NAMECOL = "WADMKC"   # << SESUAIKAN
ADMIN_KEL_NAMECOL = "WADMKD"   # << SESUAIKAN

# Nama layer di .gdb RBI
L_DESAKEL   = "ADMINISTRASI_AR_DESAKEL"     # batas kelurahan/desa
L_KECAMATAN = "ADMINISTRASI_AR_KECAMATAN"
L_KABKOTA   = "ADMINISTRASI_AR_KABKOTA"
L_RIVER     = "SUNGAI_LN_50K"
L_ROAD      = "JALAN_LN_50K"
L_LANDCOVER = "PENUTUPLAHAN_AR_50K"

# Nama kolom
COL_KABKOTA   = "WADMKK"     # nama kab/kota di layer administrasi
COL_KECAMATAN = "WADMKC"     # nama kecamatan
COL_KELURAHAN = "WADMKD"     # nama kelurahan/desa
LC_CLASS_COL  = "REMARK"     # kelas penutup lahan

# Titik banjir & non-banjir
POINTS_PER_EVENT       = 8       # titik banjir per 1 kelurahan terdampak (intensitas)
FLOOD_ELEV_QUANTILE    = 0.40    # titik banjir diambil dari elevasi <= kuantil ini (per kecamatan)
NONFLOOD_ELEV_QUANTILE = 0.50    # non-banjir dari elevasi >= median
NONFLOOD_MIN_DIST      = 750.0   # jarak min (m) dari titik banjir

NUMERIC_COLS = ["elevation", "slope", "dist_river", "dist_road", "pop_density"]
CAT_COLS     = ["landcover"]
FEATURE_COLS = NUMERIC_COLS + CAT_COLS
print("Konfigurasi dimuat. Kota:", list(CITIES))''')

# 2. Functions --------------------------------------------------------------
md("## 2. Fungsi pemrosesan geospasial")
code('''def assert_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tidak ditemukan: {path}")

def _norm(s):
    return str(s).strip().lower()

def read_gdb(gdb, layer):
    """Baca satu layer .gdb -> GeoDataFrame UTM."""
    assert_file(gdb)
    g = gpd.read_file(gdb, layer=layer)
    if g.crs is None:
        g = g.set_crs(WGS84)
    return g.to_crs(UTM_CRS)

def load_admin(path, name_col):
    """Baca file batas administrasi eksternal (.geojson/.shp/.gpkg) -> UTM,
    standarkan kolom nama jadi 'adm_name'. Memberi error jelas bila kosong/kolom salah."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File batas administrasi tidak ada: {path}\\n"
            f"  -> Sediakan batas kecamatan/kelurahan kota di folder clean/admin/.\\n"
            f"     (RBI .gdb bawaan TIDAK memuat kota target.)")
    g = gpd.read_file(path)
    if name_col not in g.columns:
        raise KeyError(f"Kolom '{name_col}' tidak ada di {os.path.basename(path)}. "
                       f"Kolom tersedia: {list(g.columns)}. Sesuaikan ADMIN_*_NAMECOL di Sel 1.")
    if g.crs is None:
        g = g.set_crs(WGS84)
    g = g.to_crs(UTM_CRS).copy()
    g["adm_name"] = g[name_col]
    return g

def city_boundary_from_admin(admin_kec_gdf):
    """Poligon kota = gabungan semua kecamatan."""
    return admin_kec_gdf.dissolve().geometry.iloc[0]

def clip_to_city(gdf, boundary_geom):
    return gdf[gdf.intersects(boundary_geom)].copy()

def clip_dem_to_bounds(dem_path, boundary_geom_wgs84, dst_crs=UTM_CRS, pad=0.02):
    """Potong DEM (lon/lat) seukuran kota + buffer, lalu reproyeksi ke UTM (hemat memori)."""
    minx, miny, maxx, maxy = boundary_geom_wgs84.bounds
    bb = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    with rasterio.open(dem_path) as src:
        arr, tr = rio_mask(src, [bb], crop=True)
        meta = src.meta.copy()
        meta.update(height=arr.shape[1], width=arr.shape[2], transform=tr)
        # reproyeksi window ke UTM
        t2, w2, h2 = calculate_default_transform(
            src.crs, dst_crs, arr.shape[2], arr.shape[1],
            *rasterio.transform.array_bounds(arr.shape[1], arr.shape[2], tr))
        dst = np.full((h2, w2), np.nan, dtype="float32")
        reproject(source=arr[0], destination=dst,
                  src_transform=tr, src_crs=src.crs,
                  dst_transform=t2, dst_crs=dst_crs, resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return dst.astype(float), t2

def compute_slope_deg(dem, transform):
    px, py = abs(transform.a), abs(transform.e)
    dzdy, dzdx = np.gradient(dem, py, px)
    return np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))

def sample_array_at_points(points_gdf, array, transform):
    vals = []
    for geom in points_gdf.geometry:
        r, c = rowcol(transform, geom.x, geom.y)
        vals.append(array[r, c] if (0 <= r < array.shape[0] and 0 <= c < array.shape[1]) else np.nan)
    return np.array(vals, dtype=float)

def attach_polygon_value(points_gdf, poly_gdf, value_col):
    poly = poly_gdf[["geometry", value_col]]
    joined = gpd.sjoin(points_gdf, poly, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.loc[points_gdf.index, value_col].values

def sample_flood_points_in_kecamatan(poly, n, dem, transform, elev_q, rng):
    """Sebar n titik banjir di dalam poligon kecamatan, hanya pada sel elevasi rendah."""
    H, W = dem.shape
    mask = rasterize([(poly, 1)], out_shape=(H, W), transform=transform, fill=0, all_touched=True).astype(bool)
    cells = mask & np.isfinite(dem)
    if cells.sum() == 0:
        return []
    thr = np.nanquantile(dem[cells], elev_q)
    low = cells & (dem <= thr)
    rows, cols = np.where(low if low.sum() >= n else cells)
    if len(rows) == 0:
        return []
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=len(rows) < n)
    pts = []
    for k in idx:
        x, y = rasterio.transform.xy(transform, rows[k], cols[k])
        pts.append(Point(x, y))
    return pts

def generate_nonflood_points(flood_gdf, dem, transform, n, elev_q, min_dist, rng):
    minx, miny, maxx, maxy = flood_gdf.total_bounds
    pad = 0.15 * max(maxx - minx, maxy - miny)
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    thr = np.nanquantile(dem, elev_q)
    flood_union = flood_gdf.unary_union
    pts, tries = [], 0
    while len(pts) < n and tries < n * 400:
        tries += 1
        px, py = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
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

print("Fungsi geospasial siap.")''')

# 3. Build per-city dataset -------------------------------------------------
md("""## 3. Bangun dataset berlabel (gabungan dua kota)
Untuk tiap kota: potong DEM, baca layer RBI, hitung kepadatan penduduk per kelurahan,
sebar titik banjir & non-banjir, lalu ekstraksi 6 faktor di tiap titik.""")
code('''def build_city_dataset(name, cfg):
    print(f"\\n========== {name} ==========")
    rng = np.random.default_rng(RANDOM_SEED)

    # --- batas administrasi (dari file eksternal) ---
    kec = load_admin(cfg["admin_kec"], ADMIN_KEC_NAMECOL)
    desakel = load_admin(cfg["admin_kel"], ADMIN_KEL_NAMECOL)
    print(f"  kecamatan={len(kec)} kelurahan={len(desakel)} (dari file admin eksternal)")

    # --- batas kota & DEM ---
    bnd_utm = city_boundary_from_admin(kec)
    bnd_wgs = gpd.GeoSeries([bnd_utm], crs=UTM_CRS).to_crs(WGS84).iloc[0]
    dem, tr = clip_dem_to_bounds(DEM_PATH, bnd_wgs)
    slope = compute_slope_deg(dem, tr)
    print("  DEM:", dem.shape, "| elev",
          round(np.nanmin(dem), 1), "-", round(np.nanmax(dem), 1), "m")

    # --- layer vektor (difilter ke kota) ---
    rivers    = clip_to_city(read_gdb(cfg["gdb"], L_RIVER), bnd_utm)
    roads     = clip_to_city(read_gdb(cfg["gdb"], L_ROAD), bnd_utm)
    landcover = clip_to_city(read_gdb(cfg["gdb"], L_LANDCOVER), bnd_utm)
    print(f"  sungai={len(rivers)} jalan={len(roads)} landcover={len(landcover)}")

    # --- kepadatan penduduk per kelurahan ---
    pen = pd.read_csv(cfg["penduduk"])
    pen["_key"] = pen["kelurahan"].map(_norm)
    desakel["_key"] = desakel["adm_name"].map(_norm)
    desakel["area_km2"] = desakel.geometry.area / 1e6
    desakel = desakel.merge(pen[["_key", "value", "value_type"]].drop_duplicates("_key"),
                            on="_key", how="left")
    is_count = desakel["value_type"].eq("count")
    desakel["pop_density"] = np.where(
        is_count, desakel["value"] / desakel["area_km2"], desakel["value"])
    print(f"  kelurahan dgn penduduk tercocokkan: "
          f"{desakel['pop_density'].notna().sum()}/{len(desakel)}")

    # --- titik banjir dari kecamatan terdampak ---
    kec["_key"] = kec["adm_name"].map(_norm)
    banjir = pd.read_csv(cfg["banjir"]); banjir["_key"] = banjir["kecamatan"].map(_norm)
    kec = kec.merge(banjir[["_key", "banjir_count"]], on="_key", how="left")
    matched = kec["banjir_count"].notna().sum()
    print(f"  kecamatan tercocok dgn data banjir: {matched}/{len(kec)}")

    flood_pts = []
    for _, row in kec.iterrows():
        cnt = row["banjir_count"]
        if pd.isna(cnt) or cnt <= 0:
            continue
        n = int(round(cnt)) * POINTS_PER_EVENT
        flood_pts += sample_flood_points_in_kecamatan(
            row.geometry, n, dem, tr, FLOOD_ELEV_QUANTILE, rng)
    flood_gdf = gpd.GeoDataFrame(geometry=flood_pts, crs=UTM_CRS)
    print(f"  titik banjir: {len(flood_gdf)}")

    # --- non-banjir 1:1 ---
    nonflood_gdf = generate_nonflood_points(
        flood_gdf, dem, tr, n=len(flood_gdf),
        elev_q=NONFLOOD_ELEV_QUANTILE, min_dist=NONFLOOD_MIN_DIST, rng=rng)
    print(f"  titik non-banjir: {len(nonflood_gdf)}")

    flood_gdf["label"] = 1; nonflood_gdf["label"] = 0
    pts = gpd.GeoDataFrame(
        pd.concat([flood_gdf[["geometry", "label"]], nonflood_gdf[["geometry", "label"]]],
                  ignore_index=True), geometry="geometry", crs=UTM_CRS)

    # --- ekstraksi 6 faktor ---
    rivers_u, roads_u = rivers.unary_union, roads.unary_union
    pts["elevation"]  = sample_array_at_points(pts, dem, tr)
    pts["slope"]      = sample_array_at_points(pts, slope, tr)
    pts["dist_river"] = pts.geometry.apply(lambda g: g.distance(rivers_u))
    pts["dist_road"]  = pts.geometry.apply(lambda g: g.distance(roads_u))
    lc_raw = attach_polygon_value(pts, landcover, LC_CLASS_COL)
    pts["landcover_raw"] = [str(v) for v in lc_raw]
    pts["pop_density"]   = attach_polygon_value(pts, desakel, "pop_density")
    pts["city"] = name

    out = pd.DataFrame(pts.drop(columns="geometry"))
    out["x"] = pts.geometry.x.values; out["y"] = pts.geometry.y.values
    # simpan layer per kota utk pemetaan nanti
    layers = dict(dem=dem, slope=slope, transform=tr, rivers=rivers, roads=roads,
                  landcover=landcover, desakel=desakel, boundary=bnd_utm)
    return out, layers

datasets, CITY_LAYERS = [], {}
for nm, cfg in CITIES.items():
    d, lyr = build_city_dataset(nm, cfg)
    datasets.append(d); CITY_LAYERS[nm] = lyr

df_all = pd.concat(datasets, ignore_index=True)

# Encoding land cover konsisten lintas kota
lc_classes = sorted(df_all["landcover_raw"].dropna().unique().tolist())
LC_MAP = {c: i for i, c in enumerate(lc_classes)}
df_all["landcover"] = df_all["landcover_raw"].map(LC_MAP).fillna(-1).astype(int)
print("\\nKelas land cover:", LC_MAP)
print("Dataset gabungan:", df_all.shape, "| distribusi:",
      df_all["label"].value_counts().to_dict())
df_all.head()''')

# 4. EDA --------------------------------------------------------------------
md("## 4. Eksplorasi singkat")
code('''fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df_all["label"].value_counts().rename({0: "Non-Banjir", 1: "Banjir"}).plot(
    kind="bar", ax=ax[0], color=["#4C9F70", "#D1495B"]); ax[0].set_title("Keseimbangan kelas")
ax[0].tick_params(axis="x", rotation=0)
corr = df_all[NUMERIC_COLS + ["label"]].corr()
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(corr))); ax[1].set_xticklabels(corr.columns, rotation=45, ha="right")
ax[1].set_yticks(range(len(corr))); ax[1].set_yticklabels(corr.columns); ax[1].set_title("Korelasi")
for (a, b), v in np.ndenumerate(corr.values):
    ax[1].text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax[1], fraction=0.046); plt.tight_layout(); plt.show()
df_all[NUMERIC_COLS].describe().round(2)''')

# 5. Preprocessing ----------------------------------------------------------
md("""## 5. Preprocessing & Feature Engineering
Buang nilai hilang/invalid, standardisasi numerik, encoding land cover.""")
code('''df = df_all.replace([np.inf, -np.inf], np.nan).copy()
before = len(df)
df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
print(f"Baris dibuang (nilai hilang/invalid): {before - len(df)} | tersisa: {len(df)}")

scaler  = StandardScaler().fit(df[NUMERIC_COLS])
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(df[CAT_COLS])

def transform_features(frame):
    Xn = pd.DataFrame(scaler.transform(frame[NUMERIC_COLS]), columns=NUMERIC_COLS, index=frame.index)
    Xc = pd.DataFrame(encoder.transform(frame[CAT_COLS]),    columns=CAT_COLS,     index=frame.index)
    return pd.concat([Xn, Xc], axis=1)

X = transform_features(df); y = df["label"].astype(int)
print("Matriks fitur:", X.shape)''')

md("## 6. Train / test split (stratified)")
code('''X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED)
print("Train:", X_train.shape, "| Test:", X_test.shape)''')

# 7. Models -----------------------------------------------------------------
md("## 7. Model individual: Random Forest, XGBoost, CatBoost")
code('''models = {}
models["Random Forest"] = RandomForestClassifier(
    n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1).fit(X_train, y_train)
models["XGBoost"] = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1).fit(X_train, y_train)
models["CatBoost"] = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05, random_seed=RANDOM_SEED, verbose=0).fit(X_train, y_train)
print("Terlatih:", list(models.keys()))''')

md("""## 8. AutoML (FLAML)
Pemilihan model + tuning hyperparameter otomatis (RF/XGBoost/CatBoost), dioptimasi ke **ROC-AUC**.""")
code('''automl = AutoML()
settings = dict(task="classification", metric="roc_auc",
                estimator_list=["rf", "xgboost", "catboost"],
                time_budget=TIME_BUDGET, eval_method="cv", n_splits=5,
                seed=RANDOM_SEED, verbose=1)
try:
    automl.fit(X_train=X_train, y_train=y_train, **settings)
except Exception as e:
    print("Ulang tanpa catboost:", e)
    settings["estimator_list"] = ["rf", "xgboost"]
    automl.fit(X_train=X_train, y_train=y_train, **settings)
print("\\nEstimator terbaik :", automl.best_estimator)
print("Konfigurasi terbaik:", automl.best_config)
models["AutoML (FLAML)"] = automl''')

# 9. Eval -------------------------------------------------------------------
md("## 9. Evaluasi")
code('''def get_proba(model, Xte):
    return np.asarray(model.predict_proba(Xte))[:, 1]

def evaluate_model(name, model, Xte, yte):
    proba = get_proba(model, Xte); pred = (proba >= 0.5).astype(int)
    return dict(Model=name, Accuracy=accuracy_score(yte, pred),
                Precision=precision_score(yte, pred, zero_division=0),
                Recall=recall_score(yte, pred, zero_division=0),
                F1=f1_score(yte, pred, zero_division=0),
                AUC_ROC=roc_auc_score(yte, proba)), proba

results, probas = [], {}
for name, m in models.items():
    metr, pr = evaluate_model(name, m, X_test, y_test); results.append(metr); probas[name] = pr
res_df = pd.DataFrame(results).set_index("Model").sort_values("AUC_ROC", ascending=False)
best_name = res_df.index[0]; best_model = models[best_name]
print("Model terbaik (AUC-ROC tertinggi):", best_name)
res_df.round(4)''')

md("### 9a. Kurva ROC")
code('''plt.figure(figsize=(7, 6))
for name, pr in probas.items():
    fpr, tpr, _ = roc_curve(y_test, pr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_test, pr):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Kurva ROC"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()''')

md("### 9b. Confusion matrix & laporan model terbaik")
code('''best_pred = (get_proba(best_model, X_test) >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-Banjir", "Banjir"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-Banjir", "Banjir"])
ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(f"Confusion Matrix — {best_name}")
for (a, b), v in np.ndenumerate(cm):
    ax.text(b, a, str(v), ha="center", va="center", color="white" if v > cm.max()/2 else "black")
plt.tight_layout(); plt.show()
print(classification_report(y_test, best_pred, target_names=["Non-Banjir", "Banjir"]))''')

md("## 10. Feature importance")
code('''frames = []
for name, m in models.items():
    fi = None
    if hasattr(m, "feature_importances_"):
        fi = np.asarray(m.feature_importances_)
    elif name == "AutoML (FLAML)":
        est = getattr(getattr(m, "model", None), "estimator", None)
        if est is not None and hasattr(est, "feature_importances_"):
            fi = np.asarray(est.feature_importances_)
    if fi is not None and len(fi) == len(FEATURE_COLS):
        frames.append(pd.Series(fi / fi.sum(), index=FEATURE_COLS, name=name))
if frames:
    imp = pd.concat(frames, axis=1); imp["mean"] = imp.mean(axis=1); imp = imp.sort_values("mean")
    imp["mean"].plot(kind="barh", figsize=(7, 4), color="#2A6F97"); plt.title("Feature importance rata-rata")
    plt.xlabel("Importance"); plt.tight_layout(); plt.show()
    display(imp.round(3))''')

# 11. Probability map -------------------------------------------------------
md("""## 11. Peta probabilitas banjir per kota (GeoTIFF + PNG)
Untuk tiap kota: bangun stack faktor selaras grid DEM, prediksi probabilitas tiap sel, simpan peta.""")
code('''def build_probability_map(name, lyr):
    dem, slope, tr = lyr["dem"], lyr["slope"], lyr["transform"]
    rivers, roads, landcover, desakel = lyr["rivers"], lyr["roads"], lyr["landcover"], lyr["desakel"]
    H, W = dem.shape; px = abs(tr.a)

    river_mask = rasterize(((g, 1) for g in rivers.geometry), out_shape=(H, W),
                           transform=tr, fill=0, all_touched=True)
    road_mask  = rasterize(((g, 1) for g in roads.geometry), out_shape=(H, W),
                           transform=tr, fill=0, all_touched=True)
    dist_river_grid = distance_transform_edt(river_mask == 0) * px
    dist_road_grid  = distance_transform_edt(road_mask == 0) * px
    lc_grid = rasterize(((g, LC_MAP.get(str(v), -1)) for g, v in
                         zip(landcover.geometry, landcover[LC_CLASS_COL])),
                        out_shape=(H, W), transform=tr, fill=-1, dtype="float32")
    pop_grid = rasterize(((g, float(v)) for g, v in
                          zip(desakel.geometry, desakel["pop_density"]) if pd.notna(v)),
                         out_shape=(H, W), transform=tr, fill=np.nan, dtype="float32")

    grid = pd.DataFrame({
        "elevation": dem.ravel(), "slope": slope.ravel(),
        "dist_river": dist_river_grid.ravel(), "dist_road": dist_road_grid.ravel(),
        "pop_density": pop_grid.ravel(), "landcover": lc_grid.ravel().astype(int)})
    valid = grid[NUMERIC_COLS].notna().all(axis=1).values
    prob = np.full(len(grid), np.nan)
    if valid.sum() > 0:
        prob[valid] = get_proba(best_model, transform_features(grid[valid]))
    prob_map = prob.reshape(H, W)

    tif = os.path.join(OUT_DIR, f"flood_probability_{name}.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs=UTM_CRS, transform=tr, nodata=np.nan) as dst:
        dst.write(prob_map.astype("float32"), 1)
    plt.figure(figsize=(8, 8))
    im = plt.imshow(prob_map, cmap="turbo", vmin=0, vmax=1)
    plt.colorbar(im, label="Probabilitas banjir")
    plt.title(f"Peta Probabilitas Banjir — {name} ({best_name})"); plt.axis("off")
    png = os.path.join(OUT_DIR, f"flood_probability_{name}.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.show()
    print("Tersimpan:", tif, "&", png)

for nm, lyr in CITY_LAYERS.items():
    build_probability_map(nm, lyr)''')

md("## 12. Simpan hasil")
code('''res_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
df.to_csv(os.path.join(OUT_DIR, "labelled_points.csv"), index=False)
print("Output di", OUT_DIR, ":", os.listdir(OUT_DIR))''')

md("""## Catatan metodologi (untuk laporan)
- **Model terbaik** dipilih dari **AUC-ROC** (sesuai paper), didampingi Precision/Recall/F1.
- **Keterbatasan data banjir:** label berasal dari BPS pada **level kecamatan** (jumlah kelurahan
  terdampak), bukan koordinat banjir presisi. Titik banjir ditempatkan secara *terrain-constrained*
  (zona elevasi rendah dalam kecamatan terdampak, jumlah ∝ intensitas). Tuliskan asumsi ini di metodologi.
- **Kepadatan penduduk Balikpapan** belum lengkap (sebagian kecamatan) sehingga sebagian titik
  ber-`pop_density` NaN dan terbuang saat preprocessing. Lengkapi `clean/penduduk_balikpapan.csv` bila ada data baru.
- DEM dipotong per kota lalu direproyeksi ke UTM 50S agar slope & jarak bersatuan meter.""")

nb["cells"] = cells
out_path = os.path.join(BASE, "Flood_AutoML_Local.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook ditulis:", out_path, "|", len(cells), "sel")
