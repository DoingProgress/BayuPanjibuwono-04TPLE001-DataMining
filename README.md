# Data Mining - Analisis Pelayanan Pasien & Klasifikasi Iris
**BayuPanjibuwono-04TPLE001-DataMining**

Repository ini berisi 3 proyek Data Mining untuk Tugas 04TPLE001, mulai dari cleaning data klinik, klasifikasi spesies bunga Iris, sampai exploratory data analysis (EDA).

### Daftar Isi
1. [klinik.py - Analisis Pelayanan Pasien](#1-klinikpy---analisis-pelayanan-pasien)
2. [iris.py - Klasifikasi Bunga Iris](#2-irispy---klasifikasi-bunga-iris)
3. [datmin_pt6.py - EDA & Data Quality Check](#3-datmin_pt6py---eda--data-quality-check)
4. [Cara Menjalankan](#cara-menjalankan)

---

### 1. klinik.py - Analisis Pelayanan Pasien
File utama untuk dataset `Dataset_Pelayanan_Pasien.csv`.
### Tujuan
Melakukan evaluasi kilas balik (retrospective evaluation) terhadap data pelayanan pasien klinik untuk mengidentifikasi distribusi kategori umur, 
pola 10 diagnosa terbanyak, perbandingan pasien lama dan baru, serta beban pelayanan per dokter sebagai dasar evaluasi kinerja dan pengambilan keputusan manajemen klinik.

**Fitur:**
- **Cleaning:** Ekstrak angka tahun dari format "14 thn 7 bln", trim Diagnosa, isi missing value Dokter & Status Pasien.
- **Klasifikasi Umur:** Fungsi `kategorikan_umur()` -> Anak-anak (<18), Dewasa (<60), Lansia (>=60). Divisualisasikan dengan `DecisionTreeClassifier(max_depth=3)` dan `LabelEncoder`.
- **Rekapitulasi:** 
    - Kategori Umur (Jumlah & %)
    - 10 Diagnosa Terbanyak
    - Status Pasien Lama vs Baru
    - Jumlah Pasien per Dokter
- **Visualisasi:** 4 grafik dalam 1 figure + 1 Decision Tree Kategori Umur.
- **Output:** `grafik_klinik.png`, `grafik_decision_tree_klinik.png`

### 2. iris.py - Klasifikasi Bunga Iris
Menganalisis 2 file sekaligus: `Iris_dataset.csv` dan `Iris.csv`.

**Fitur Canggih:**
- **Pencarian File Case-Insensitive:** Fungsi `cari_file()` agar tidak error beda huruf besar/kecil.
- **Cleaning Robust:** Normalisasi nama kolom dengan regex, konversi numerik `coerce`, bersihkan prefix `Iris-` pada species.
- **Collecting:** Menggabungkan kedua file, tetap simpan info sumber, hitung baris tidak valid yang dihapus.
- **Analisis:** Jumlah & persentase per spesies, cross-tab per file, rata-rata ukuran sepal/petal per spesies.
- **Klasifikasi:** `train_test_split(stratify=y)`, `RandomForestClassifier(n_estimators=200)` untuk akurasi, dan `DecisionTreeClassifier` terpisah untuk visualisasi aturan. Evaluasi dengan `accuracy_score`, `classification_report`, `confusion_matrix`.
- **Visualisasi:** Bar jumlah spesies, Pie persentase, Scatter petal length vs width, dan Confusion Matrix.
- **Output:** `grafik_iris.png`, `grafik_decision_tree.png`

### 3. datmin_pt6.py - EDA & Data Quality Check
Fokus untuk Laporan Data Quality sesuai modul Data Mining.

**Fitur:**
- **Ringkasan Dataset:** Jumlah baris, kolom, `head()`.
- **Pemisahan Tipe Data:** Deteksi otomatis `numeric_cols` via `select_dtypes` dan `categorical_cols`.
- **Data Quality:** Hitung `duplicated()`, `isnull()`, `eq(0)` untuk cek zero, plus rincian per kolom.
- **Pandas Functions:** Demonstrasi fungsi yang diminta: `read_csv`, `select_dtypes`, `isnull`, `duplicated`, `value_counts`, `fillna`, `astype`, `eq`, `sum`, `groupby`, `sort_values`, `head`, `to_string`.
- **Encoding Contoh:** `LabelEncoder` untuk kolom `Jk` (Jenis Kelamin).
- **Decision Tree Status Pasien:** Klasifikasi `Status Pasien` berdasarkan `Umur_Tahun` dan `Jk_Code` dengan `class_weight="balanced"`.
- **Visualisasi:** Grafik Missing Value per kolom & Decision Tree Status Pasien.




### Cara Menjalankan
bash
# 1. Clone repo baru
git clone https://github.com/DoingProgress/BayuPanjibuwono-04TPLE001-DataMining.git
cd BayuPanjibuwono-04TPLE001-DataMining

# 2. Buat venv
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate # Mac/Linux

pip install pandas numpy matplotlib scikit-learn

# 3. Jalankan satu per satu
python klinik.py
python iris.py
python datmin_pt6.py
