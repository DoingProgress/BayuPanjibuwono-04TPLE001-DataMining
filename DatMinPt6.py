from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

FOLDER = Path(__file__).resolve().parent
DATASET_PATH = FOLDER / "Dataset_Pelayanan_Pasien.csv"


def main():
    df = pd.read_csv(DATASET_PATH)
    umur_tahun = pd.to_numeric(
        df["Umur"].astype("string").str.extract(r"(\d+)\s*thn", expand=False),
        errors="coerce",
    )

    print("1. RINGKASAN DATASET")
    print(f"Jumlah baris: {len(df)}")
    print(f"Jumlah kolom: {len(df.columns)}")
    print(df.head().to_string(index=False))

    print("\n2. JENIS DATA")
    numeric_data = pd.DataFrame({"Umur (tahun)": umur_tahun})
    numeric_cols = numeric_data.columns.tolist()
    categorical_cols = sorted(
        column for column in df.columns if column != "Umur"
    )

    print(f"2.1 Data numerikal ({len(numeric_cols)} kolom):")
    for nomor, col in enumerate(numeric_cols, start=1):
        print(f"  {nomor}. {col}: {numeric_data[col].dtype}")

    print(f"\n2.2 Data kategorikal ({len(categorical_cols)} kolom):")
    for nomor, col in enumerate(categorical_cols, start=1):
        jumlah_kategori = int(df[col].nunique(dropna=True))
        print(f"  {nomor}. {col}: {jumlah_kategori} nilai unik")

    print("\n3. JUMLAH DUPLIKAT, NULL, DAN ZERO")
    duplicate_count = int(df.duplicated().sum())
    null_total = int(df.isnull().sum().sum())
    zero_per_kolom = numeric_data.eq(0).sum().sort_values(ascending=False)
    zero_total = int(np.count_nonzero(numeric_data.to_numpy() == 0))

    print(f"3.1 Jumlah baris duplikat: {duplicate_count}")
    print(f"3.2 Jumlah nilai null: {null_total}")
    print(f"3.3 Jumlah nilai zero pada kolom numerikal: {zero_total}")

    print("\n3.4 Rincian null per kolom:")
    null_per_kolom = df.isnull().sum().sort_values(ascending=False)
    null_per_kolom = null_per_kolom[null_per_kolom > 0]
    for nomor, (kolom, jumlah) in enumerate(null_per_kolom.items(), start=1):
        print(f"  {nomor}. {kolom}: {jumlah}")
    if null_per_kolom.empty:
        print("  Tidak ada nilai null.")

    print("\n3.5 Rincian zero per kolom numerikal:")
    for nomor, (kolom, jumlah) in enumerate(zero_per_kolom.items(), start=1):
        print(f"  {nomor}. {kolom}: {jumlah}")
    print("  Catatan: umur 0 tahun berarti usia masih di bawah 1 tahun.")

    # Contoh pengkodean kategorikal dengan LabelEncoder dari scikit-learn
    if "Jk" in df.columns:
        le = LabelEncoder()
        df["Jk_Code"] = le.fit_transform(df["Jk"].astype(str))
        print("\n4. CONTOH LABEL ENCODING JENIS KELAMIN")
        print(df[["Jk", "Jk_Code"]].head().to_string(index=False))

    # Pohon keputusan eksploratif untuk mengklasifikasikan status pasien.
    data_pohon = pd.DataFrame(
        {
            "Umur_Tahun": umur_tahun,
            "Jk_Code": df["Jk_Code"],
            "Status Pasien": df["Status Pasien"],
        }
    ).dropna()
    encoder_status = LabelEncoder()
    target_status = encoder_status.fit_transform(data_pohon["Status Pasien"])
    model_pohon = DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=42,
    )
    model_pohon.fit(data_pohon[["Umur_Tahun", "Jk_Code"]], target_status)

    fig_pohon, ax_pohon = plt.subplots(figsize=(14, 8))
    plot_tree(
        model_pohon,
        feature_names=["Umur (tahun)", "Jenis kelamin (kode)"],
        class_names=encoder_status.classes_.tolist(),
        filled=True,
        rounded=True,
        impurity=False,
        ax=ax_pohon,
    )
    ax_pohon.set_title("Decision Tree untuk Klasifikasi Status Pasien")
    fig_pohon.tight_layout()
    file_grafik_pohon = FOLDER / "grafik_decision_tree_datminpt6.png"
    fig_pohon.savefig(file_grafik_pohon, dpi=180, bbox_inches="tight")
    print(f"\n5. Grafik decision tree disimpan ke: {file_grafik_pohon}")
    plt.close(fig_pohon)

    # Visualisasi missing values
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_counts.plot(kind="bar", color="#D65F5F", ax=ax)
        ax.set_title("Jumlah Missing Value per Kolom")
        ax.set_xlabel("Kolom")
        ax.set_ylabel("Jumlah Missing")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        file_grafik = FOLDER / "grafik_missing_values.png"
        fig.savefig(file_grafik, dpi=150, bbox_inches="tight")
        print(f"\n6. Grafik missing values disimpan ke: {file_grafik}")
        plt.close(fig)
    else:
        print("\n6. Tidak ada missing value untuk divisualisasikan.")


if __name__ == "__main__":
    main()
