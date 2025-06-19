# Laporan Proyek Machine Learning Terapan 1: Predictive-Analytics-Student-Performance-Dataset - Ahmad Farhan Juliansyah

## Domain Proyek
Proyek ini berfokus pada prediksi `Performance Index` siswa menggunakan data yang mencakup waktu belajar, skor sebelumnya, kegiatan ekstrakurikuler, jam tidur, dan latihan soal. Ini bertujuan untuk memprediksi `Performance Index` siswa berdasarkan beberapa faktor yang dianggap mempengaruhi kinerja akademik, seperti jumlah jam belajar, skor sebelumnya, jam tidur, kegiatan ekstrakurikuler, dan jumlah soal yang dipraktikkan. Proyek ini dapat membantu untuk mengembangkan model prediksi yang dapat membantu sekolah atau lembaga pendidikan dalam meramalkan kinerja akademik siswa dan merencanakan intervensi yang tepat. 
Beberapa penelitian terkait yang mendasari pemilihan variabel dalam proyek ini adalah:
- [Factors affecting high school students’ academic performance: a case study in Vietnam] https://www.researchgate.net/publication/392576622_Factors_affecting_high_school_students%27_academic_performance_a_case_study_in_Vietnam
- [Sleep quality, duration, and consistency are associated with better academic performance in college students] https://www.nature.com/articles/s41539-019-0055-z?utm_source=chatgpt.com
- [Determining factors that affect student performance using various machine learning methods] https://www.sciencedirect.com/science/article/pii/S1877050922022529

## Business Understanding
### Problem Statements
1. **Bagaimana kita dapat memprediksi kinerja akademik siswa (`Performance Index`) berdasarkan fitur-fitur yang tersedia?**
2. **Bagaimana kita dapat meningkatkan akurasi prediksi kinerja siswa menggunakan beberapa algoritma machine learning?**

### Goals
1. **Membangun model yang dapat memprediksi `Performance Index` siswa dengan menggunakan data yang melibatkan waktu belajar, skor sebelumnya, kegiatan ekstrakurikuler, dan jam tidur.**
2. **Mengevaluasi dan membandingkan kinerja tiga model machine learning: Support Vector Machine (SVM), Random Forest, dan K-Nearest Neighbors (KNN) untuk memilih model yang paling baik menggunakan metrik MSE (Mean Squared Error).**
  
### Solution statements
Untuk mencapai tujuan di atas, solusi yang diajukan adalah:
1. **Menggunakan beberapa algoritma machine learning (SVM, Random Forest, KNN) untuk membandingkan kinerjanya dalam memprediksi `Performance Index`.**
2. **Metrik evaluasi menggunakan Mean Squared Error (MSE), selanjutnya memilih model yang paling bagus berdasarkan nilai metrik tersebut.**

## Data Understanding
### Kondisi dataset:
1. Dataset yang digunakan dalam proyek ini adalah **Student Performance Dataset** yang berisi 10.000 entri dengan 6 fitur yang mencakup data numerik dan kategorikal.
2. Dataset ini tersedia di Kaggle dengan link: [dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
3. Dataset tidak ada nilai null atau missing value.
4. Dataset terdapat nilai duplikasi sebanyak 127 baris dan penanganan selanjutkan akan menghapus 127 baris yang terduplikasi untuk menjaga keakuratan dataset. Jumlah baris dataset menjadi 9.873.
5. Variabel target dalam dataset ini adalah `Performance Index`.


### Fitur-fitur pada dataset ini adalah sebagai berikut:
- `Hours Studied`: Jumlah jam yang dihabiskan siswa untuk belajar.
- `Previous Scores`: Skor yang telah dicapai siswa sebelumnya.
- `Extracurricular Activities`: Kegiatan ekstrakurikuler yang diikuti siswa dan satu-satunya fitur yang jenisnya kategorikal.
- `Sleep Hours`: Jumlah jam tidur siswa.
- `Sample Question Papers Practiced`: Jumlah soal latihan yang telah dikerjakan siswa.
- `Performance Index`: Nilai kinerja akademik yang ingin diprediksi dan fitur ini yang dijadikan data targetnya.

### **Exploratory Data Analysis**:
1. Terdapat kolom kategorikal yang perlu diproses melalui encoding. Kolom kategorikal dalam dataset ini adalah Extracurricular Activities. Kolom ini menyimpan data berupa kategori (apakah siswa mengikuti kegiatan ekstrakurikuler atau tidak) dan perlu diubah menjadi bentuk numerik agar bisa digunakan dalam pemodelan machine learning.
2. Korelasi antara fitur-fitur numerik menunjukkan beberapa hubungan yang signifikan.
3. Fitur target `Performance Index` menunjukkan variasi yang cukup baik yang dapat digunakan dalam model prediksi.

## Data Preparation

### Informasi Dataset
Untuk melihat informasi dasar mengenai dataset, seperti jumlah kolom, tipe data pada setiap kolom, dan jumlah data non-null per kolom.
```
df.info()
```

### Deskripsi Statistik
Menggunakan fungsi `describe()` untuk melihat deskripsi statistik dataset, yang mencakup nilai rata-rata, standar deviasi, nilai minimum, dan maksimum pada setiap kolom numerik.
```
df.describe()
```

### Nilai Null
Mengecek apakah ada nilai null pada dataset dan menghitung jumlahnya.
```
df.isnull().sum()
```

### Drop Duplicate value
Data duplikat dihapus untuk memastikan kualitas dataset yang lebih baik. Proses pembersihan data dilakukan dengan menghapus data duplikat yang dapat mempengaruhi hasil analisis dan pemodelan.
```
df.duplicated().sum()
df_cleaned = df.drop_duplicates()
```

### Outliers
Pada tahap ini dicek menggunakan boxplot apakah data pada kolom Hours Studied, Previous Scores, Sleep Hours, Sample Question Papers Practiced, dan Performance Index ada outlier atau semua nilai berada dalam rentang yang wajar

### Missing Value
Dataset dicek apakah ada missing values menggunakan fungsi, dan jika ada bisa diatasi dengan mengisi atau menghapusnya.
```
df_cleaned.isna().sum()
```

### Encoding Fitur Kategori
Kolom `Extracurricular Activities` di-encode menggunakan Label Encoding untuk mengubah nilai kategorikal menjadi numerik.

### Train Test Split
Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan fungsi `train_test_split` dari `sklearn`, untuk memastikan bahwa model dapat dievaluasi dengan data yang belum pernah dilihat sebelumnya.

### Standarisasi
Fitur numerik seperti `Hours Studied`, `Previous Scores`, `Sleep Hours`, dan `Sample Question Papers Practiced` distandarisasi menggunakan `StandardScaler` untuk memastikan setiap fitur memiliki distribusi dengan mean = 0 dan varians = 1, sehingga meningkatkan kinerja model.


## Modeling

### SVM

**Parameter** :
- Kernel: 'rbf'
- C: 1.0
- Epsilon: 0.1

**Kelebihan**:
- Dapat menangani hubungan non-linear antara fitur dan target dengan baik.
- Cocok untuk dataset yang lebih kecil atau medium.

**Kekurangan**:
- Memerlukan tuning hyperparameter yang cukup banyak.
- Dapat lambat pada dataset yang besar.

### Random Forest Regressor

**Parameter**:
- n_estimators: 50
- max_depth: 16
- random_state: 55
- n_jobs: 1

**Kelebihan**:
- Dapat menangani dataset besar dan sangat efektif dalam menangani outliers.
- Dapat memberikan estimasi pentingnya fitur dalam model.

**Kekurangan**:
- Lebih lambat dalam prediksi dibandingkan model linear.
- Rentan terhadap overfitting jika tidak diatur dengan baik.

### K-Nearest Neighbors

**Parameter**:
- n_neighbors: 20

**Kelebihan**:
- Sangat sederhana dan mudah dipahami.
- Tidak memerlukan asumsi tentang distribusi data.

**Kekurangan**:
- Sensitif terhadap data outlier.
- Performa buruk pada dataset yang sangat besar.

### **Pemilihan Model**
Dari tiga model yang dievaluasi, **Random Forest** menunjukkan hasil terbaik dengan MSE yang lebih rendah, baik pada data pelatihan maupun data uji. Model ini lebih stabil dan akurat dibandingkan dengan SVM dan KNN.

## Evaluation
**Metrik Evaluasi yang Digunakan:**
- **Mean Squared Error (MSE)** digunakan untuk mengukur kesalahan prediksi dari model yang dibangun.

**Pembahasan Hasil**
Model **Random Forest** menunjukkan hasil terbaik dengan MSE yang rendah baik pada data pelatihan maupun pengujian, sehingga dipilih sebagai model terbaik untuk prediksi `Performance Index` siswa. Meskipun **KNN** memberikan hasil yang cukup baik, namun **Random Forest** lebih unggul dalam hal keakuratan.

Kesimpulannya, **Random Forest** adalah model yang paling sesuai untuk memprediksi kinerja siswa berdasarkan fitur yang tersedia, dengan tingkat kesalahan yang minimal.
