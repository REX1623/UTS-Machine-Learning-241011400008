# UTS Machine Learning — Linear & Logistic Regression dari Nol

Implementasi **Linear Regression** dan **Logistic Regression** menggunakan **NumPy** (tanpa library ML), lalu dibandingkan dengan scikit-learn. Proyek ini dikerjakan sebagai tugas Ujian Tengah Semester mata kuliah Machine Learning.

| | |
| --- | --- |
| **Nama** | Muhammad Syafryan Rahmadhani |
| **NIM** | 241011400008 |
| **Kelas** | Reg B |
| **Random State** | 8 (dua digit terakhir NIM) |

---

## Daftar Isi

- [Struktur Proyek](#struktur-proyek)
- [Dataset](#dataset)
- [Implementasi](#implementasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Hasil](#hasil)
- [Kesimpulan](#kesimpulan)

---

## Struktur Proyek

```
.
├── UTS_Machine_Learning.ipynb   # Notebook utama (kode, hasil, dan analisis)
├── Ads_Data.csv                 # Dataset regresi (biaya iklan vs penjualan)
├── Email_Data.csv               # Dataset klasifikasi (deteksi spam)
└── README.md
```

## Dataset

### `Ads_Data.csv` — Regresi

Hubungan antara biaya iklan dan penjualan (dalam juta rupiah). Total **40 baris**.

| Kolom | Tipe | Keterangan |
| --- | --- | --- |
| `BiayaIklan` | float | Biaya iklan (juta Rp) — **fitur** |
| `Penjualan` | float | Nilai penjualan (juta Rp) — **target** |

### `Email_Data.csv` — Klasifikasi

Fitur sederhana email untuk mendeteksi spam. Total **60 baris** (35 bukan spam, 25 spam).

| Kolom | Tipe | Keterangan |
| --- | --- | --- |
| `JumlahLink` | int | Jumlah link di dalam email — **fitur** |
| `KataPromo` | int | Jumlah kata promosi — **fitur** |
| `Spam` | int | Label: `0` = bukan spam, `1` = spam — **target** |

## Implementasi

Semua algoritma ditulis manual di notebook:

| Komponen | Deskripsi |
| --- | --- |
| `train_test_split` | Pembagian data latih/uji dengan `np.random.permutation` |
| `LinearRegression` | Solusi analitik via **Normal Equation** |
| `LinearRegressionGD` | Linear Regression dengan **Gradient Descent** (loss: MSE) |
| `LogisticRegression` | Klasifikasi biner dengan **Gradient Descent** (loss: log loss / cross-entropy) |
| Metrik regresi | MAE, MSE, RMSE, R² |
| Metrik klasifikasi | Confusion matrix, Accuracy, Precision, Recall, F1-Score |

### Alur Pengerjaan

**Bagian A — Linear Regression (Ads_Data)**

1. Eksplorasi data dan scatter plot
2. Split data 80:20
3. Training dengan Normal Equation dan interpretasi koefisien
4. Evaluasi (MAE, MSE, RMSE, R²) pada data training dan testing
5. Eksperimen *learning rate* Gradient Descent (0.001, 0.02, 0.1) beserta kurva loss
6. Visualisasi garis regresi dan prediksi biaya iklan baru (4.5, 7, dan 12 juta)

**Bagian B — Logistic Regression (Email_Data)**

1. Eksplorasi data dan scatter plot per kelas
2. Split data 75:25, training dengan Gradient Descent (lr = 0.1, 10.000 iterasi)
3. Kurva log loss
4. Evaluasi: confusion matrix, accuracy, precision, recall, F1
5. Visualisasi decision boundary
6. Analisis pengaruh threshold (0.3, 0.5, 0.7)
7. Prediksi 4 email baru

**Bagian C — Pertanyaan konseptual** dan **Bonus**: perbandingan dengan scikit-learn.

## Cara Menjalankan

### Prasyarat

- Python 3.9+
- Library: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (scikit-learn hanya untuk bagian bonus), `jupyter`

### Instalasi

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Menjalankan

```bash
jupyter notebook UTS_Machine_Learning.ipynb
```

Pastikan `Ads_Data.csv` dan `Email_Data.csv` berada di **folder yang sama** dengan notebook, lalu jalankan semua cell secara berurutan (*Run All*).

## Hasil

### Linear Regression (Normal Equation)

Persamaan regresi:

```
Penjualan = 28.9945 + 9.0688 × BiayaIklan
```

Setiap kenaikan biaya iklan 1 juta rupiah diprediksi menaikkan penjualan sekitar 9,07 juta rupiah.

| Data | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- |
| Training | 4.2685 | 28.2836 | 5.3182 | 0.9493 |
| Testing | 5.3610 | 68.6741 | 8.2870 | 0.9154 |

**Eksperimen learning rate (Gradient Descent):**

| Learning Rate | Intercept | Koefisien | MSE Akhir | Keterangan |
| --- | --- | --- | --- | --- |
| 0.001 | 25.994 | 9.566 | 30.309 | Terlalu kecil, belum konvergen penuh |
| 0.02 | 28.995 | 9.069 | 28.284 | Konvergen, sama dengan Normal Equation |
| 0.1 | ~3.5 × 10⁶ | ~2.1 × 10⁷ | ~1.4 × 10¹⁶ | Terlalu besar, divergen |

**Prediksi biaya iklan baru:**

| Biaya Iklan (juta) | Prediksi Penjualan (juta) | Catatan |
| --- | --- | --- |
| 4.5 | 69.80 | Interpolasi |
| 7.0 | 92.48 | Interpolasi |
| 12.0 | 137.82 | Ekstrapolasi (di luar rentang data, kurang andal) |

### Logistic Regression

Persamaan model:

```
P(Spam) = σ(-8.4338 + 0.7612 × JumlahLink + 0.7255 × KataPromo)
```

Performa pada data uji (15 sampel, threshold 0.5):

| Metrik | Nilai |
| --- | --- |
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |

Confusion matrix: `[[12, 0], [0, 3]]`

**Pengaruh threshold:**

| Threshold | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| 0.3 | 0.9333 | 0.75 | 1.00 |
| 0.5 | 1.0000 | 1.00 | 1.00 |
| 0.7 | 1.0000 | 1.00 | 1.00 |

**Prediksi email baru:**

| Email | JumlahLink | KataPromo | P(Spam) | Prediksi |
| --- | --- | --- | --- | --- |
| E1 | 1 | 2 | 0.0020 | Normal |
| E2 | 5 | 6 | 0.4317 | Normal |
| E3 | 9 | 10 | 0.9966 | Spam |
| E4 | 2 | 11 | 0.7445 | Spam |

### Perbandingan dengan scikit-learn

| Model | Implementasi Manual | scikit-learn |
| --- | --- | --- |
| Linear Regression | b = 28.9945, m = 9.0688 | b = 28.9945, m = 9.0688 |
| Logistic Regression | b = -8.4338, m = [0.7612, 0.7255] | b = -8.6030, m = [0.7767, 0.7391] |

Linear Regression menghasilkan koefisien identik. Pada Logistic Regression terdapat selisih kecil karena perbedaan metode optimasi (Gradient Descent murni vs solver L-BFGS).

## Kesimpulan

- Biaya iklan dan penjualan memiliki hubungan linear yang kuat (R² uji ≈ 0.92).
- Normal Equation dan Gradient Descent dengan learning rate yang tepat (0.02) menghasilkan solusi yang sama; learning rate yang terlalu besar menyebabkan divergen, yang terlalu kecil menyebabkan konvergensi lambat.
- Logistic Regression memisahkan email spam dan non-spam dengan sangat baik pada data uji. Perlu diingat bahwa data uji hanya 15 sampel, sehingga skor sempurna belum tentu mencerminkan performa pada data yang lebih besar.
- Pada filter spam, *false positive* (email penting masuk ke folder spam) umumnya lebih merugikan daripada *false negative*, sehingga threshold yang lebih tinggi (0.5–0.7) lebih disarankan daripada 0.3.
- Implementasi manual dari nol terbukti menghasilkan hasil yang sangat dekat dengan scikit-learn.
