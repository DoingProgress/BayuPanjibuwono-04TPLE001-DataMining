from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# 1. LOADING DATA
folder_script = Path(__file__).resolve().parent
df = pd.read_csv(folder_script / "Dataset_Pelayanan_Pasien.csv")

# 2. CLEANING & PREPROCESSING DATA
# Mengambil angka tahun dari string kolom 'Umur' (misal: "14 thn 7 bln" -> 14)
df["Umur_Tahun"] = (
    df["Umur"].astype(str).str.extract(r"(\d+)\s*thn").fillna(0).astype(int)
)

# Menghilangkan spasi berlebih pada Diagnosa
df["Diagnosa_Clean"] = (
    df["Diagnosa"].fillna("TIDAK ADA DIAGNOSA").astype(str).str.strip()
)

# Menangani missing values pada Dokter dan Status Pasien
df["Dokter"] = df["Dokter"].fillna("Tidak Diketahui")
df["Status Pasien"] = df["Status Pasien"].fillna("Tidak Diketahui")


# 3. KLASIFIKASI KATEGORI UMUR
def kategorikan_umur(umur):
    if umur < 18:
        return "Anak-anak"
    elif umur < 60:
        return "Dewasa"
    else:
        return "Lansia"


df["Kategori_Umur"] = df["Umur_Tahun"].apply(kategorikan_umur)

# Pohon keputusan memvisualisasikan aturan kategori umur berdasarkan Umur_Tahun.
model_pohon_umur = DecisionTreeClassifier(max_depth=3, random_state=42)
model_pohon_umur.fit(df[["Umur_Tahun"]], df["Kategori_Umur"])

# Menggunakan LabelEncoder dari Scikit-Learn untuk enkoding numerik
le = LabelEncoder()
df["Kategori_Umur_Code"] = le.fit_transform(df["Kategori_Umur"])

# 4. REKAPITULASI DATA & PERSENTASE
# a. Kategori Umur
umur_counts = df["Kategori_Umur"].value_counts()
umur_pct = (df["Kategori_Umur"].value_counts(normalize=True) * 100).round(2)
rekap_umur = pd.DataFrame({"Jumlah": umur_counts, "Persentase (%)": umur_pct})

# b. 10 Diagnosa Terbanyak
diag_counts = df["Diagnosa_Clean"].value_counts().head(10)
diag_pct = (df["Diagnosa_Clean"].value_counts(normalize=True).head(10) * 100).round(2)
rekap_diag = pd.DataFrame({"Jumlah": diag_counts, "Persentase (%)": diag_pct})

# c. Perbandingan Pasien Lama dan Baru
status_counts = df["Status Pasien"].value_counts()
status_pct = (df["Status Pasien"].value_counts(normalize=True) * 100).round(2)
rekap_status = pd.DataFrame({"Jumlah": status_counts, "Persentase (%)": status_pct})

# d. Jumlah Pasien Berdasarkan Nama Dokter
dokter_counts = df["Dokter"].value_counts()
dokter_pct = (df["Dokter"].value_counts(normalize=True) * 100).round(2)
rekap_dokter = pd.DataFrame({"Jumlah": dokter_counts, "Persentase (%)": dokter_pct})

# Menampilkan Hasil Analisis di Terminal
print("=== REKAPITULASI KATEGORI UMUR ===")
print(rekap_umur)
print("\n=== 10 DIAGNOSA TERBANYAK ===")
print(rekap_diag)
print("\n=== STATUS PASIEN (LAMA VS BARU) ===")
print(rekap_status)
print("\n=== JUMLAH PASIEN PER DOKTER ===")
print(rekap_dokter)

# 5. VISUALISASI DATA DENGAN MATPLOTLIB
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafik 1: Kategori Umur
axes[0, 0].bar(
    rekap_umur.index,
    rekap_umur["Jumlah"],
    color=["#4C72B0", "#55A868", "#C44E52"],
)
axes[0, 0].set_title("Distribusi Kategori Umur Pasien")
axes[0, 0].set_ylabel("Jumlah Pasien")

# Grafik 2: Status Pasien (Pie Chart)
axes[0, 1].pie(
    rekap_status["Jumlah"],
    labels=rekap_status.index,
    autopct="%1.1f%%",
    colors=["#6ACC65", "#4878D0"],
    startangle=90,
)
axes[0, 1].set_title("Perbandingan Pasien Lama vs Baru")

# Grafik 3: Pasien per Dokter
axes[1, 0].barh(rekap_dokter.index, rekap_dokter["Jumlah"], color="#8172B3")
axes[1, 0].set_title("Jumlah Pasien Berdasarkan Dokter")
axes[1, 0].set_xlabel("Jumlah Pasien")
axes[1, 0].invert_yaxis()

# Grafik 4: 10 Diagnosa Terbanyak
axes[1, 1].barh(rekap_diag.index, rekap_diag["Jumlah"], color="#CCB974")
axes[1, 1].set_title("10 Diagnosa / Penyakit Terbanyak")
axes[1, 1].set_xlabel("Jumlah Pasien")
axes[1, 1].invert_yaxis()

plt.tight_layout()
file_grafik = folder_script / "grafik_klinik.png"
fig.savefig(file_grafik, dpi=150, bbox_inches="tight")
print(f"\nGrafik berhasil disimpan ke: {file_grafik}")
plt.show()

fig_pohon, ax_pohon = plt.subplots(figsize=(14, 8))
plot_tree(
    model_pohon_umur,
    feature_names=["Umur_Tahun"],
    class_names=model_pohon_umur.classes_.tolist(),
    filled=True,
    rounded=True,
    impurity=False,
    ax=ax_pohon,
)
ax_pohon.set_title("Decision Tree Kategori Umur Pasien")
fig_pohon.tight_layout()
file_grafik_pohon = folder_script / "grafik_decision_tree_klinik.png"
fig_pohon.savefig(file_grafik_pohon, dpi=180, bbox_inches="tight")
print(f"Grafik decision tree disimpan ke: {file_grafik_pohon}")
plt.show()
