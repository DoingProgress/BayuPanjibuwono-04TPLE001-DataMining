from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


FOLDER_SCRIPT = Path(__file__).resolve().parent
FITUR = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
NAMA_FITUR = {
    "sepallengthcm": "sepal_length_cm",
    "sepalwidthcm": "sepal_width_cm",
    "petallengthcm": "petal_length_cm",
    "petalwidthcm": "petal_width_cm",
    "species": "species",
}


def cari_file(nama: str) -> Path:
    """Cari file dengan memperhitungkan perbedaan huruf besar/kecil."""
    kandidat = [FOLDER_SCRIPT / nama]
    kandidat.extend(
        path
        for path in FOLDER_SCRIPT.iterdir()
        if path.is_file() and path.name.lower() == nama.lower() and path not in kandidat
    )
    for path in kandidat:
        if path.is_file():
            return path
    raise FileNotFoundError(f"File '{nama}' tidak ditemukan di {FOLDER_SCRIPT}")


def muat_dan_bersihkan(path: Path) -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(path)
    nama_kolom = {
        re.sub(r"[^a-z0-9]", "", str(kolom).lower()): kolom
        for kolom in data.columns
    }

    kolom_diperlukan = list(NAMA_FITUR)
    kolom_hilang = [kolom for kolom in kolom_diperlukan if kolom not in nama_kolom]
    if kolom_hilang:
        raise ValueError(
            f"Kolom {kolom_hilang} tidak ditemukan pada file '{path.name}'. "
            f"Kolom yang tersedia: {list(data.columns)}"
        )

    data = data.rename(
        columns={
            nama_kolom[kolom_asal]: kolom_tujuan
            for kolom_asal, kolom_tujuan in NAMA_FITUR.items()
        }
    )
    data = data[FITUR + ["species"]].copy()

    for fitur in FITUR:
        data[fitur] = pd.to_numeric(data[fitur], errors="coerce")
    data["species"] = (
        data["species"]
        .astype("string")
        .str.strip()
        .str.replace(r"^iris[-_\s]*", "", regex=True, case=False)
        .str.lower()
    )

    nilai_fitur = data[FITUR].to_numpy(dtype=float)
    baris_valid = np.isfinite(nilai_fitur).all(axis=1)
    baris_valid &= data["species"].notna().to_numpy()
    baris_valid &= data["species"].ne("").to_numpy()
    jumlah_dihapus = int((~baris_valid).sum())
    data = data.loc[baris_valid].copy()
    return data, jumlah_dihapus


def cetak_ringkasan_kualitas(path: Path) -> None:
    """Tampilkan klasifikasi kolom dan kualitas data sebelum pembersihan."""
    data = pd.read_csv(path)
    kolom_numerikal = data.select_dtypes(include=[np.number]).columns.tolist()
    kolom_identitas = [
        kolom
        for kolom in kolom_numerikal
        if re.sub(r"[^a-z0-9]", "", str(kolom).lower()) in {"id", "index"}
    ]
    kolom_numerikal = sorted(
        kolom for kolom in kolom_numerikal if kolom not in kolom_identitas
    )
    kolom_kategorikal = sorted(
        data.select_dtypes(exclude=[np.number]).columns.tolist() + kolom_identitas
    )

    print(f"=== JENIS DATA DAN KUALITAS DATA: {path.name} ===")
    print(f"Jumlah baris: {len(data)}")
    print(f"Jumlah kolom: {len(data.columns)}")

    print(f"\nKolom numerikal ({len(kolom_numerikal)}):")
    for kolom in kolom_numerikal:
        print(f"  {kolom}: {data[kolom].dtype}")

    print(f"\nKolom kategorikal ({len(kolom_kategorikal)}):")
    for kolom in kolom_kategorikal:
        jumlah_unik = int(data[kolom].nunique(dropna=True))
        print(f"  {kolom}: {data[kolom].dtype}, {jumlah_unik} nilai unik")

    duplikat = int(data.duplicated().sum())
    jumlah_null = int(data.isna().sum().sum())
    jumlah_zero_per_kolom = data[kolom_numerikal].eq(0).sum().sort_values(
        ascending=False
    )
    jumlah_zero = int(np.count_nonzero(data[kolom_numerikal].to_numpy() == 0))

    print("\nDuplikat, null, dan zero:")
    print(f"  Baris duplikat: {duplikat}")
    print(f"  Nilai null: {jumlah_null}")
    print(f"  Nilai zero pada kolom numerikal: {jumlah_zero}")

    null_per_kolom = data.isna().sum().sort_values(ascending=False)
    null_per_kolom = null_per_kolom[null_per_kolom > 0]
    print("  Rincian null per kolom:")
    if null_per_kolom.empty:
        print("    Tidak ada nilai null.")
    else:
        for kolom, jumlah in null_per_kolom.items():
            print(f"    {kolom}: {jumlah}")

    print("  Rincian zero per kolom numerikal:")
    for kolom, jumlah in jumlah_zero_per_kolom.items():
        print(f"    {kolom}: {int(jumlah)}")


# 1. LOADING DATA
file_iris = cari_file("Iris.csv")
print("=== PEMUATAN DATA ===")
cetak_ringkasan_kualitas(file_iris)

data_sumber = []
for nama_sumber, path in [("Iris.csv", file_iris)]:
    data, jumlah_dihapus = muat_dan_bersihkan(path)
    data["sumber"] = nama_sumber
    data_sumber.append(data)
    print(
        f"{nama_sumber}: {len(data)} baris valid "
        f"({jumlah_dihapus} baris tidak valid dihapus)"
    )

# Gabungkan kedua file; catatan pengukuran berulang tetap dipertahankan.
df = pd.concat(data_sumber, ignore_index=True)
print(f"Jumlah data dianalisis: {len(df)} baris\n")

# 2. COLLECTING / RINGKASAN DATA
jumlah_spesies = df["species"].value_counts().sort_index()
persentase_spesies = (
    df["species"].value_counts(normalize=True).sort_index().mul(100).round(2)
)
rekap_spesies = pd.DataFrame(
    {"Jumlah": jumlah_spesies, "Persentase (%)": persentase_spesies}
)

print("=== JUMLAH DAN PERSENTASE BERDASARKAN SPESIES ===")
print(rekap_spesies)

rekap_sumber_spesies = pd.crosstab(df["sumber"], df["species"])
print("\n=== JUMLAH SPESIES PER FILE ===")
print(rekap_sumber_spesies)

print("\n=== RATA-RATA UKURAN BUNGA PER SPESIES ===")
print(df.groupby("species")[FITUR].mean().round(2))

# 3. KLASIFIKASI SPESIES DENGAN SCIKIT-LEARN
jumlah_per_kelas = df["species"].value_counts()
if len(jumlah_per_kelas) < 2 or jumlah_per_kelas.min() < 2:
    raise ValueError(
        "Klasifikasi membutuhkan minimal dua spesies, "
        "dengan setidaknya dua baris valid untuk setiap spesies."
    )

X = df[FITUR].to_numpy(dtype=float)
y = df["species"].to_numpy()
X_latih, X_uji, y_latih, y_uji = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_latih, y_latih)
prediksi = model.predict(X_uji)

# Pohon keputusan terpisah agar struktur aturan klasifikasi dapat divisualisasikan.
model_pohon = DecisionTreeClassifier(max_depth=3, random_state=42)
model_pohon.fit(X_latih, y_latih)

akurasi = accuracy_score(y_uji, prediksi)
print("\n=== HASIL KLASIFIKASI SPESIES ===")
print(f"Jumlah data latih: {len(y_latih)}")
print(f"Jumlah data uji: {len(y_uji)}")
print(f"Akurasi: {akurasi:.2%}")
print(
    classification_report(
        y_uji,
        prediksi,
        labels=model.classes_,
        zero_division=0,
    )
)

# 4. VISUALISASI DAN SIMPAN HASIL
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
warna = {"setosa": "#4C72B0", "versicolor": "#55A868", "virginica": "#C44E52"}

rekap_spesies["Jumlah"].plot(
    kind="bar",
    ax=axes[0, 0],
    color=[warna.get(spesies, "#8172B3") for spesies in rekap_spesies.index],
)
axes[0, 0].set_title("Jumlah Data per Spesies")
axes[0, 0].set_xlabel("Spesies")
axes[0, 0].set_ylabel("Jumlah")
axes[0, 0].tick_params(axis="x", rotation=0)

rekap_spesies["Persentase (%)"].plot(
    kind="pie",
    ax=axes[0, 1],
    autopct="%1.1f%%",
    startangle=90,
    colors=[warna.get(spesies, "#8172B3") for spesies in rekap_spesies.index],
)
axes[0, 1].set_title("Persentase Data per Spesies")
axes[0, 1].set_ylabel("")

for spesies, kelompok in df.groupby("species"):
    axes[1, 0].scatter(
        kelompok["petal_length_cm"],
        kelompok["petal_width_cm"],
        label=spesies,
        alpha=0.7,
        color=warna.get(spesies, "#8172B3"),
    )
axes[1, 0].set_title("Panjang dan Lebar Petal")
axes[1, 0].set_xlabel("Panjang petal (cm)")
axes[1, 0].set_ylabel("Lebar petal (cm)")
axes[1, 0].legend(title="Spesies")

matriks_konfusi = confusion_matrix(y_uji, prediksi, labels=model.classes_)
gambar_konfusi = axes[1, 1].imshow(matriks_konfusi, cmap="Blues")
axes[1, 1].set_title("Confusion Matrix")
axes[1, 1].set_xlabel("Prediksi")
axes[1, 1].set_ylabel("Aktual")
axes[1, 1].set_xticks(range(len(model.classes_)), model.classes_, rotation=30)
axes[1, 1].set_yticks(range(len(model.classes_)), model.classes_)
for baris in range(matriks_konfusi.shape[0]):
    for kolom in range(matriks_konfusi.shape[1]):
        axes[1, 1].text(
            kolom,
            baris,
            str(matriks_konfusi[baris, kolom]),
            ha="center",
            va="center",
        )
fig.colorbar(gambar_konfusi, ax=axes[1, 1], label="Jumlah")

fig.tight_layout()
file_grafik = FOLDER_SCRIPT / "grafik_iris.png"
fig.savefig(file_grafik, dpi=150, bbox_inches="tight")
print(f"Grafik disimpan ke: {file_grafik}")

fig_pohon, ax_pohon = plt.subplots(figsize=(18, 10))
plot_tree(
    model_pohon,
    feature_names=FITUR,
    class_names=model_pohon.classes_.tolist(),
    filled=True,
    rounded=True,
    impurity=False,
    ax=ax_pohon,
)
ax_pohon.set_title("Decision Tree untuk Klasifikasi Spesies Iris (maks. kedalaman 3)")
fig_pohon.tight_layout()
file_grafik_pohon = FOLDER_SCRIPT / "grafik_decision_tree.png"
fig_pohon.savefig(file_grafik_pohon, dpi=180, bbox_inches="tight")
print(f"Grafik decision tree disimpan ke: {file_grafik_pohon}")
plt.show()
