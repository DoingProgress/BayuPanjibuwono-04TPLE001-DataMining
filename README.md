Data Mining - Analisis Pelayanan Pasien & Klasifikasi Iris
*BayuPanjibuwono-04TPLE001-DataMining*

Repository ini berisi 3 proyek Data Mining untuk Tugas 04TPLE001, mulai dari cleaning data klinik, klasifikasi spesies bunga Iris, sampai exploratory data analysis (EDA).

Daftar Isi
1. klinik.py - Analisis Pelayanan Pasien
2. iris.py - Klasifikasi Bunga Iris
3. DatMinPt6.py - EDA & Data Quality Check
4. Cara Menjalankan

Struktur File.
.
├── venv
├── .gitignore
├── Dataset_Pelayanan_Pasien.csv
├── DatMinPt6.py
├── klinik.py
├── iris.py
├── Iris.csv
├── iris_dataset.csv
├── grafik_klinik.png
├── grafik_decision_tree_klinik.png
├── grafik_iris.png
├── grafik_decision_tree.png
├── grafik_decision_tree_datminpt6.png
├── grafik_missing_values.png
├── README.md
└── requirements.txt

1. klinik.py - Analisis Pelayanan Pasien
File untuk `Dataset_Pelayanan_Pasien.csv`.

*Tujuan:* Evaluasi retrospektif pelayanan pasien klinik untuk distribusi umur, 10 diagnosa terbanyak, pasien lama vs baru, dan beban per dokter.

*Fitur:*
- Cleaning: Ekstrak tahun dari "14 thn 7 bln", trim Diagnosa, fillna Dokter & Status Pasien
- Klasifikasi Umur: `kategorikan_umur()` -> Anak-anak (<18), Dewasa (<60), Lansia (>=60) + `DecisionTreeClassifier(max_depth=3)` dan `LabelEncoder`
- Rekap: Kategori Umur (Jumlah & %), 10 Diagnosa Terbanyak, Status Pasien, Jumlah per Dokter
- Output: `grafik_klinik.png`, `grafik_decision_tree_klinik.png`

2. iris.py - Klasifikasi Bunga Iris
Menganalisis `Iris.csv` dan `iris_dataset.csv`.

*Fitur:*
- Pencarian File Case-Insensitive: `cari_file()`
- Cleaning Robust: Normalisasi kolom pakai regex, `pd.to_numeric(coerce)`, bersihkan prefix `Iris-`
- Collecting: Gabung 2 file, simpan info sumber, hitung baris tidak valid
- Analisis: Jumlah & persentase per spesies, crosstab per file, rata-rata sepal/petal
- Klasifikasi: `train_test_split(stratify=y)`, `RandomForestClassifier(n_estimators=200)`, `DecisionTreeClassifier`, `accuracy_score`, `classification_report`, `confusion_matrix`
- Output: `grafik_iris.png`, `grafik_decision_tree.png`

3. DatMinPt6.py - EDA & Data Quality Check
Fokus Laporan Data Quality.

*Fitur:*
- Ringkasan: baris, kolom, `head()`
- Tipe Data: `select_dtypes` untuk numeric & kategorikal
- Data Quality: `duplicated()`, `isnull()`, `eq(0)`, rincian per kolom
- Fungsi Pandas: `read_csv`, `select_dtypes`, `isnull`, `duplicated`, `value_counts`, `fillna`, `astype`, `eq`, `sum`, `groupby`, `sort_values`, `head`, `to_string`
- Encoding: `LabelEncoder` untuk `Jk`
- Decision Tree Status Pasien: `class_weight="balanced"`
- Output: `grafik_missing_values.png`, `grafik_decision_tree_datminpt6.png`

Cara Menjalankan
git clone https://github.com/DoingProgress/BayuPanjibuwono-04TPLE001-DataMining.git
cd BayuPanjibuwono-04TPLE001-DataMining

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python klinik.py
python iris.py
python DatMinPt6.py
pandas
numpy
matplotlib
scikit-learn

Author
Bayu Panjibuwono - 04TPLE001 - Data Mining
