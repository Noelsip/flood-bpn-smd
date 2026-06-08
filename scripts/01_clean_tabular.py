"""
01_clean_tabular.py
===================
Membersihkan data tabular mentah (banjir BPS + penduduk) menjadi file rapi
yang dipakai notebook Flood_AutoML_Local.ipynb.

Output (folder ./clean):
  - banjir_balikpapan.csv   kolom: kota, kecamatan, banjir_count
  - banjir_samarinda.csv    kolom: kota, kecamatan, banjir_count
  - penduduk_balikpapan.csv kolom: kota, kecamatan, kelurahan, value, value_type
  - penduduk_samarinda.csv  kolom: kota, kecamatan, kelurahan, value, value_type

Catatan:
  * banjir_count = jumlah kelurahan/desa terdampak banjir per KECAMATAN
    (Balikpapan: maksimum dari tahun 2019-2021; Samarinda: tahun 2025).
  * value_type penduduk: "count"  = jumlah jiwa  -> notebook hitung density = value/luas
                         "density" = jiwa/km2     -> notebook pakai langsung.
  * Penduduk Balikpapan HANYA 3/6 kecamatan (Barat=count, Utara & Kota=density).
    Kecamatan lain akan NaN sampai datanya dilengkapi.

Jalankan:  python scripts/01_clean_tabular.py
"""
import os
import re
import glob
import unicodedata
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "clean")
os.makedirs(OUT_DIR, exist_ok=True)


def clean_name(s):
    """Normalisasi nama wilayah: buang kode [..]/(..), spasi & karakter aneh."""
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\[[^\]]*\]", "", s)      # buang [010]
    s = re.sub(r"\([^)]*\)", "", s)       # buang (001)
    s = re.sub(r"<[^>]+>", "", s)         # buang tag html
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in ("", "nan", "none"):
        return None
    return s


def to_number(v):
    """Ubah nilai ke angka. Angka asli (dari Excel) dikembalikan apa adanya;
    string '-', '–', kosong -> None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and pd.isna(v)) else float(v)
    s = str(v).strip().replace("\xa0", "").replace("–", "-")
    if s in ("", "-", "nan", "None"):
        return None
    try:
        return float(s)            # mis. "5030" / "2311.53"
    except ValueError:
        s2 = s.replace(",", "")    # buang pemisah ribuan gaya Inggris "1,234"
        try:
            return float(s2)
        except ValueError:
            return None


# ----------------------------------------------------------------------------
# 1. BANJIR SAMARINDA  (banjir/Banjir_samarinda.csv)
#    Baris data mulai setelah header berlapis; kolom 0=kecamatan, 1=Banjir(2025).
# ----------------------------------------------------------------------------
def clean_banjir_samarinda():
    path = os.path.join(BASE, "banjir", "Banjir_samarinda.csv")
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    rows = []
    for _, r in raw.iterrows():
        kec = clean_name(r[0])
        val = to_number(r[1]) if len(r) > 1 else None
        if not kec or val is None:
            continue
        if kec.lower() in ("kecamatan", "samarinda", "banjir"):  # header / total kota
            continue
        rows.append({"kota": "Kota Samarinda", "kecamatan": kec,
                     "banjir_count": int(val)})
    df = pd.DataFrame(rows).drop_duplicates("kecamatan")
    out = os.path.join(OUT_DIR, "banjir_samarinda.csv")
    df.to_csv(out, index=False)
    print(f"[banjir samarinda] {len(df)} kecamatan -> {out}")
    return df


# ----------------------------------------------------------------------------
# 2. BANJIR BALIKPAPAN  (banjir/Banjir_balikpapan.csv)
#    Kolom: kecamatan, Banjir 2019, 2020, 2021, Longsor 2019.. -> ambil MAX banjir.
# ----------------------------------------------------------------------------
def clean_banjir_balikpapan():
    path = os.path.join(BASE, "banjir", "Banjir_balikpapan.csv")
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    rows = []
    for _, r in raw.iterrows():
        kec = clean_name(r[0])
        if not kec or not kec.lower().startswith("balikpapan "):
            continue  # hanya baris "Balikpapan <Kecamatan>", buang total & header
        if kec.lower() == "balikpapan":   # total kota
            continue
        vals = [to_number(r[i]) for i in (1, 2, 3)]   # banjir 2019,2020,2021
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        rows.append({"kota": "Kota Balikpapan", "kecamatan": kec,
                     "banjir_count": int(max(vals))})
    df = pd.DataFrame(rows).drop_duplicates("kecamatan")
    out = os.path.join(OUT_DIR, "banjir_balikpapan.csv")
    df.to_csv(out, index=False)
    print(f"[banjir balikpapan] {len(df)} kecamatan -> {out}")
    return df


# ----------------------------------------------------------------------------
# 3. PENDUDUK SAMARINDA  (padat-penduduk/Samarinda/*.csv)
#    Baris kecamatan punya kode [0x0]; baris kelurahan menjorok + kode (00x).
#    Ambil kolom terakhir (Laki+Perempuan = jumlah penduduk) untuk kelurahan.
# ----------------------------------------------------------------------------
def clean_penduduk_samarinda():
    path = glob.glob(os.path.join(BASE, "padat-penduduk", "Samarinda", "*.csv"))[0]
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    rows, cur_kec = [], None
    for _, r in raw.iterrows():
        rawname = str(r[0])
        if not rawname.strip():
            continue
        name = clean_name(rawname)
        total = to_number(r[len(r) - 1])  # kolom terakhir = L+P
        is_kec = bool(re.search(r"\[\d+\]", rawname))     # punya [010]
        is_kel = bool(re.search(r"\(\d+\)", rawname))     # punya (001)
        if is_kec:
            cur_kec = name
        elif is_kel and total is not None and cur_kec:
            rows.append({"kota": "Kota Samarinda", "kecamatan": cur_kec,
                         "kelurahan": name, "value": total, "value_type": "count"})
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "penduduk_samarinda.csv")
    df.to_csv(out, index=False)
    print(f"[penduduk samarinda] {len(df)} kelurahan / "
          f"{df['kecamatan'].nunique()} kecamatan -> {out}")
    return df


# ----------------------------------------------------------------------------
# 4. PENDUDUK BALIKPAPAN  (padat-penduduk/Balikpapan/*.xlsx) -- 3 file, unit campur
#    Deteksi otomatis: judul mengandung "Kepadatan" -> density, "Jumlah" -> count.
#    Nama kecamatan diambil dari judul sel A1 ("... Kecamatan Balikpapan Barat").
# ----------------------------------------------------------------------------
def clean_penduduk_balikpapan():
    files = sorted(glob.glob(os.path.join(BASE, "padat-penduduk", "Balikpapan", "*.xlsx")))
    rows = []
    for f in files:
        df0 = pd.read_excel(f, header=None, dtype=object)
        title = " ".join(str(x) for x in df0.iloc[0].tolist() if x is not None)
        m = re.search(r"Kecamatan\s+(Balikpapan\s+\w+)", title)
        kec = clean_name(m.group(1)) if m else clean_name(df0.iloc[0, 0])
        is_density = "kepadatan" in title.lower()
        vtype = "density" if is_density else "count"

        # Cari baris data: nama di kolom 0, dan ADA angka di kolom 1..n.
        # Untuk file 'count' (Barat): kolom terakhir = Jumlah 2023.
        # Untuk file 'density': kolom terakhir = density 2023.
        for i in range(1, len(df0)):
            name = clean_name(df0.iloc[i, 0])
            if not name or name.lower().startswith(("catatan", "sumber", "laki", "perempuan", "jumlah")):
                continue
            if name.lower() == kec.lower():       # baris total kecamatan -> lewati
                continue
            # ambil angka kolom paling kanan yang tidak kosong
            vals = [to_number(v) for v in df0.iloc[i, 1:].tolist()]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            val = vals[-1]   # tahun terbaru (2023) ada di paling kanan
            rows.append({"kota": "Kota Balikpapan", "kecamatan": kec,
                         "kelurahan": name, "value": val, "value_type": vtype})
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "penduduk_balikpapan.csv")
    df.to_csv(out, index=False)
    covered = sorted(df["kecamatan"].unique())
    print(f"[penduduk balikpapan] {len(df)} kelurahan / {len(covered)} kecamatan "
          f"({', '.join(covered)}) -> {out}")
    print("   !! Balikpapan hanya sebagian kecamatan. Lengkapi sisanya bila ada.")
    return df


if __name__ == "__main__":
    print("=== Membersihkan data tabular ===")
    clean_banjir_samarinda()
    clean_banjir_balikpapan()
    clean_penduduk_samarinda()
    clean_penduduk_balikpapan()
    print("Selesai. Cek folder:", OUT_DIR)
