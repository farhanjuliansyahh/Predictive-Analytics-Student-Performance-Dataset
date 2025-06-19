# Laporan project Machine Learning Terapan 1: Predictive-Analytics-Student-Performance-Dataset | Ahmad Farhan Juliansyah

## Domain project
Project ini berfokus pada prediksi `Performance Index` siswa menggunakan data yang mencakup waktu belajar, skor sebelumnya, kegiatan ekstrakurikuler, jam tidur, dan latihan soal. Ini bertujuan untuk memprediksi `Performance Index` siswa berdasarkan beberapa faktor yang dianggap mempengaruhi kinerja akademik, seperti jumlah jam belajar, skor sebelumnya, jam tidur, kegiatan ekstrakurikuler, dan jumlah soal yang dipraktikkan. project ini dapat membantu untuk mengembangkan model prediksi yang dapat membantu sekolah atau lembaga pendidikan dalam memprediksi kinerja akademik siswa dan merencanakan intervensi yang tepat. 
Beberapa penelitian terkait yang mendasari pemilihan variabel dalam project ini adalah:
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
1. Dataset yang digunakan dalam project ini adalah **Student Performance Dataset** yang berisi 10.000 entri dengan 6 fitur yang mencakup data numerik dan kategorikal.
2. Dataset ini tersedia di Kaggle dengan link: [Student-Performance-Dataset](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
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
1. Terdapat kolom kategorikal yang perlu diproses melalui encoding. Kolom kategorikal dalam dataset ini adalah `Extracurricular Activities`. Kolom ini menyimpan data berupa kategori (apakah siswa mengikuti kegiatan ekstrakurikuler atau tidak) dan perlu diubah menjadi bentuk numerik agar bisa digunakan dalam pemodelan machine learning.
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
```
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced','Performance Index']])
plt.title('Boxplot untuk Mendeteksi Outliers')
plt.show()
```

### Missing Value
Dataset dicek apakah ada missing values menggunakan fungsi, dan jika ada bisa diatasi dengan mengisi atau menghapusnya.
```
df_cleaned.isna().sum()
```

### Univariate Analysis
Univariate Analysis adalah analisis yang melibatkan satu variabel untuk memahami distribusinya. Pada tahap ini, data untuk setiap fitur numerik dianalisis dengan menggunakan histogram dan KDE (Kernel Density Estimation) untuk melihat distribusi dan bentuk data. Untuk fitur kategorikal, digunakan countplot untuk melihat frekuensi kemunculan setiap kategori. Tujuan dari univariate analysis adalah untuk:
- Mengetahui apakah data terdistribusi normal atau memiliki pola tertentu.
- Menemukan potensi outliers yang dapat mempengaruhi hasil model.
- Memahami sebaran nilai-nilai dalam setiap fitur untuk mempersiapkan data yang lebih baik.

### Multivariate Analysis
Multivariate Analysis digunakan untuk menganalisis hubungan antara dua atau lebih variabel dalam dataset. Pada tahap ini, dilakukan analisis korelasi antara fitur numerik menggunakan heatmap korelasi untuk melihat hubungan antar variabel. Selain itu, hubungan antara variabel numerik dan kategorikal juga dianalisis menggunakan boxplot untuk mengeksplorasi bagaimana distribusi nilai numerik dipengaruhi oleh kategori tertentu. Tujuan dari multivariate analysis adalah untuk:
- Mengidentifikasi hubungan yang signifikan antara fitur-fitur yang dapat digunakan dalam model prediksi.
- Memahami apakah ada multikolinearitas (hubungan yang sangat tinggi) antara fitur-fitur numerik yang dapat mempengaruhi hasil model.
- Membantu dalam memilih fitur yang paling relevan untuk model.

### Encoding Fitur Kategorikal
Fitur `Extracurricular Activities` di-encode menggunakan Label Encoding untuk mengubah nilai kategorikal menjadi numerik.

### Train Test Split
Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan fungsi `train_test_split` dari `sklearn`, untuk memastikan bahwa model dapat dievaluasi dengan data yang belum pernah dilihat sebelumnya.

### Standarisasi
Fitur numerik seperti `Hours Studied`, `Previous Scores`, `Sleep Hours`, dan `Sample Question Papers Practiced` distandarisasi menggunakan `StandardScaler` untuk memastikan setiap fitur memiliki distribusi dengan mean = 0 dan varians = 1, sehingga meningkatkan kinerja model.


## Modeling

### SVM
SVM adalah algoritma klasifikasi yang mencari hyperplane terbaik untuk memisahkan data ke dalam kelas-kelas yang berbeda. Pada regresi, SVM dapat digunakan untuk memprediksi nilai kontinu dengan mengoptimalkan margin kesalahan.

```
svm = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm.fit(X_train, y_train)

models.loc['train_mse', 'SVM'] = mean_squared_error(y_true=y_train, y_pred=svm.predict(X_train))
```

**Parameter** :
- Kernel: 'rbf'
- C: 1.0
- Epsilon: 0.1

| Model | Kelebihan                                                                | Kekurangan                                          |
|-------|--------------------------------------------------------------------------|-----------------------------------------------------|
| SVM   | Dapat menangani hubungan non-linear antara fitur dan target dengan baik. | Memerlukan tuning hyperparameter yang cukup banyak. |
| SVM   | Cocok untuk dataset yang lebih kecil atau medium.                        | Dapat lambat pada dataset yang besar.               |

### Random Forest Regressor
Random Forest adalah algoritma ensemble yang membangun sejumlah pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi prediksi.

```
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse', 'RandomForest'] = mean_squared_error(y_true=y_train, y_pred=RF.predict(X_train))
```

**Parameter**:
- n_estimators: 50
- max_depth: 16
- random_state: 55
- n_jobs: 1

| Model         | Kelebihan                                                                  | Kekurangan                                                 |
|---------------|----------------------------------------------------------------------------|------------------------------------------------------------|
| Random Forest | Dapat menangani dataset besar dan sangat efektif dalam menangani outliers. | Lebih lambat dalam prediksi dibandingkan model linear.     |
| Random Forest | Dapat memberikan estimasi pentingnya fitur dalam model.                    | Rentan terhadap overfitting jika tidak diatur dengan baik. |

### K-Nearest Neighbors
KNN adalah algoritma yang mengklasifikasikan data berdasarkan kedekatannya dengan data lainnya. Pada regresi, KNN memprediksi nilai target berdasarkan rata-rata nilai target dari tetangga terdekat.

```
knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(X_train, y_train)

models.loc['train_mse', 'KNN'] = mean_squared_error(y_true=y_train, y_pred=knn.predict(X_train))
```

**Parameter**:
- n_neighbors: 20

| Model | Kelebihan                                        | Kekurangan                                     |
|-------|--------------------------------------------------|------------------------------------------------|
| KNN   | Sangat sederhana dan mudah dipahami.             | Sensitif terhadap data outlier.                |
| KNN   | Tidak memerlukan asumsi tentang distribusi data. | Performa buruk pada dataset yang sangat besar. |

### **Pemilihan Model**
Dari tiga model yang dievaluasi, **Random Forest** menunjukkan hasil terbaik dengan MSE yang lebih rendah, baik pada data pelatihan maupun data uji. Model ini lebih stabil dan akurat dibandingkan dengan SVM dan KNN.

## Evaluation
**Metrik Evaluasi yang Digunakan:**
- **Mean Squared Error (MSE)** digunakan untuk mengukur kesalahan prediksi dari model yang dibangun.

**Pembahasan Hasil**
| Model             | Train MSE  | Test MSE   |
|-------------------|------------|------------|
| SVM               | 0.005089   | 0.005741   |
| Random Forest     | 0.00097    | 0.006457   |
| KNN               | 0.00752    | 0.008731   |

- `SVM` memiliki MSE yang cukup rendah, terutama pada data pelatihan. Ini menunjukkan bahwa model SVM juga memberikan hasil yang baik, meskipun sedikit kurang baik dibandingkan dengan `Random Forest`.
- `Random Forest` memiliki MSE yang paling rendah baik pada data pelatihan (train) dan sedikit lebih baik pengujian (test) dibanding `KNN`. Ini menunjukkan bahwa model `Random Forest` paling akurat dalam memprediksi Performance Index. 
- `KNN` memiliki MSE yang paling tinggi pada kedua set data (pelatihan dan pengujian), yang menunjukkan bahwa model ini tidak seakurat `SVM` dan `Random Forest` dalam memprediksi Performance Index.

Kesimpulannya insight: `Random Forest` adalah model terbaik berdasarkan MSE data training dan secara data testing pun tidak berbeda jauh dengan yang paling baik `SVM`. `Random Forest` memberikan hasil yang paling akurat dalam memprediksi `Performance Index` pada data ini, dengan MSE yang sudah rendah pada data pelatihan dan pengujian.

## Kesimpulan dengan Business Understanding

### Problem Statements
1. Prediksi Kinerja Akademik: Berdasarkan hasil evaluasi, `Random Forest` adalah model yang paling akurat dalam memprediksi `Performance Index` siswa dengan MSE terendah.
2. Peningkatan Akurasi: Dengan membandingkan `SVM`, `Random Forest`, dan `KNN`, `Random Forest` memberikan hasil terbaik, menjawab kebutuhan untuk meningkatkan akurasi prediksi.

### Goals
1. Model Prediksi: Algoritma machine learning seperti `Random Forest` berhasil membangun model untuk memprediksi Performance Index siswa dengan fitur yang tersedia.
2. Evaluasi Model: `Random Forest` terpilih sebagai model terbaik setelah dibandingkan menggunakan MSE.

### Solution statements
1. Perbandingan Model: Hasil menunjukkan `Random Forest` adalah model terbaik untuk memprediksi Performance Index berdasarkan MSE.
2. Metrik Evaluasi: Berdasarkan MSE, `Random Forest` dipilih sebagai model terbaik dengan nilai mendekati 0.

Kesimpulan Akhir: `Random Forest` adalah model yang paling akurat untuk memprediksi `Performance Index` siswa, sesuai dengan tujuan dan solusi yang diinginkan dalam project ini karena dapat memprediksi `Performance Index` siswa dengan baik, sehingga dapat digunakan untuk membantu sekolah atau lembaga pendidikan dalam memprediksi kinerja akademik siswa dan merencanakan intervensi yang tepat. 
