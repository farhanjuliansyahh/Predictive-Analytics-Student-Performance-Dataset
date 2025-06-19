# Laporan Proyek Machine Learning Terapan 1: Predictive-Analytics-Student-Performance-Dataset - Ahmad Farhan Juliansyah

## Domain Proyek
Proyek ini berfokus pada prediksi `Performance Index` siswa menggunakan data yang mencakup waktu belajar, skor sebelumnya, kegiatan ekstrakurikuler, jam tidur, dan latihan soal. Ini bertujuan untuk memprediksi `Performance Index` siswa berdasarkan beberapa faktor yang dianggap mempengaruhi kinerja akademik, seperti jumlah jam belajar, skor sebelumnya, jam tidur, kegiatan ekstrakurikuler, dan jumlah soal yang dipraktikkan. Proyek ini dapat membantu untuk mengembangkan model prediksi yang dapat membantu sekolah atau lembaga pendidikan dalam meramalkan kinerja akademik siswa dan merencanakan intervensi yang tepat. Beberapa penelitian terkait yang mendasari pemilihan variabel dalam proyek ini adalah
[Factors affecting high school students’ academic performance: a case study in Vietnam] https://www.researchgate.net/publication/392576622_Factors_affecting_high_school_students%27_academic_performance_a_case_study_in_Vietnam
[Sleep quality, duration, and consistency are associated with better academic performance in college students] https://www.nature.com/articles/s41539-019-0055-z?utm_source=chatgpt.com
[Determining factors that affect student performance using various machine learning methods] https://www.sciencedirect.com/science/article/pii/S1877050922022529

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
2. **Melakukan optimasi model melalui tuning hyperparameter jika diperlukan, untuk meningkatkan akurasi prediksi.**

## Data Understanding
### Kondisi dataset:
Dataset yang digunakan dalam proyek ini adalah **Student Performance Dataset** yang berisi 10.000 entri dengan 6 fitur yang mencakup data numerik dan kategorikal. Variabel target dalam dataset ini adalah `Performance Index`. Dataset ini tersedia di Kaggle. 

### Variabel-variabel pada dataset ini adalah sebagai berikut:
- `Hours Studied`: Jumlah jam yang dihabiskan siswa untuk belajar.
- `Previous Scores`: Skor yang telah dicapai siswa sebelumnya.
- `Extracurricular Activities`: Kegiatan ekstrakurikuler yang diikuti siswa (kategorikal).
- `Sleep Hours`: Jumlah jam tidur siswa.
- `Sample Question Papers Practiced`: Jumlah soal latihan yang telah dikerjakan siswa.
- `Performance Index`: Nilai kinerja akademik yang ingin diprediksi (target).

### **Exploratory Data Analysis**:
1. Data memiliki beberapa outliers yang perlu diwaspadai.
2. Terdapat kolom kategorikal yang perlu diproses melalui encoding.
3. Korelasi antara fitur-fitur numerik menunjukkan beberapa hubungan yang signifikan.
4. Variabel target `Performance Index` menunjukkan variasi yang cukup baik yang dapat digunakan dalam model prediksi.

## Data Preparation

### Drop Duplicate value
Data duplikat dihapus untuk memastikan kualitas dataset yang lebih baik.

### Encoding Fitur Kategori
Kolom `Extracurricular Activities` di-encode menggunakan Label Encoding untuk mengubah nilai kategorikal menjadi numerik.

### Train Test Split
Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan fungsi `train_test_split` dari `sklearn`.

### Standarisasi
Fitur numerik seperti `Hours Studied`, `Previous Scores`, `Sleep Hours`, dan `Sample Question Papers Practiced` distandarisasi menggunakan `StandardScaler` untuk memastikan setiap fitur memiliki distribusi dengan mean = 0 dan varians = 1.

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
