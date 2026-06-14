ini adalah dataset dan program ipynb yang ku punya.
tapi ada revisi dari dosen yaitu sebagai berikut:
perlu dibuat time series dan berapa lama yg ingin direncanakan, apakah 1 hari, 1 Minggu atau gmn
ntuk meramalkan 7 hari kedepan, perlu menentukan data harian aja atau kemarin atau satu minggu sebelumnya
data perlu di transformasi
dt-1 d-2 smpai dt-6
tentukan dulu berapa lama rentan yg ingin di ramalkan

Dari komentar dosen, masalah utama bukan di model AutoML-mu, tetapi pada cara membentuk data time series.

Saat ini dataset dan notebook-mu masih cenderung menggunakan fitur cuaca pada hari yang sama untuk memprediksi banjir pada hari yang sama. Untuk peramalan (forecasting), dosen meminta pendekatan berbasis lag time series.

Yang perlu kamu tentukan terlebih dahulu:

Horizon Prediksi (Forecast Horizon)

Pertanyaan dosen:

"berapa lama yang ingin diramalkan?"

Pilihan yang umum:

Prediksi 1 hari ke depan → Flood(t+1)
Prediksi 3 hari ke depan → Flood(t+3)
Prediksi 7 hari ke depan → Flood(t+7)

Karena banjir biasanya tidak bisa diprediksi terlalu jauh dengan akurat menggunakan data cuaca harian, saya sarankan:

Forecast 7 hari ke depan

karena sesuai dengan revisi dosen dan lebih menarik untuk penelitian.

Target menjadi:

Flood
t+7
	​


Artinya:

Data cuaca hari ini digunakan untuk memprediksi apakah akan terjadi banjir 7 hari kemudian.

Transformasi Time Series

Komentar dosen:

dt-1, dt-2 sampai dt-6

Artinya kamu harus membuat fitur lag.

Misalnya variabel:

rain_sum
precipitation_sum
soil_moisture

diubah menjadi:

tanggal	rain_t-1	rain_t-2	rain_t-3	rain_t-4	rain_t-5	rain_t-6

Contoh:

Tanggal 10 Januari:

fitur	nilai
rain_t-1	hujan 9 Januari
rain_t-2	hujan 8 Januari
rain_t-3	hujan 7 Januari
rain_t-4	hujan 6 Januari
rain_t-5	hujan 5 Januari
rain_t-6	hujan 4 Januari

Ini yang dimaksud transformasi time series.

Bentuk Dataset yang Disarankan

Misalnya:

Target:

Flood
t+7
	​


Feature:

rain_t-1
rain_t-2
rain_t-3
rain_t-4
rain_t-5
rain_t-6
soil_t-1
soil_t-2
soil_t-3
soil_t-4
soil_t-5
soil_t-6
tempmax_t-1
tempmax_t-2
dst

Target:

Flood_t+7

Contoh:

waktu	rain_t-1	rain_t-2	soil_t-1	soil_t-2	Target
2024-01-10	25	15	0.42	0.40	Flood 17 Jan
Split Data Harus Time-Aware

Untungnya notebook-mu sudah melakukan:

ordered = df_train.sort_values("time")
cut_idx = int(len(ordered) * 0.70)

train_df = ordered.iloc[:cut_idx]
test_df = ordered.iloc[cut_idx:]

Ini sudah benar untuk time series.

Jangan menggunakan random split lagi.

Yang Saya Sarankan untuk Skripsi/Tugas Akhir Ini

Karena dataset-mu:

2016–2025
6452 data
56 kejadian banjir

Saya sarankan:

Forecast horizon = 7 hari

Transformasi:

dt-1
dt-2
dt-3
dt-4
dt-5
dt-6
dt-7

Target:

Flood
t+7
	​


Sehingga narasi metodologi menjadi:

Data cuaca harian ditransformasikan menggunakan metode lag feature dengan jendela observasi 7 hari sebelumnya (t−1 sampai t−7). Variabel target didefinisikan sebagai kejadian banjir pada hari ke-t+7 sehingga model dapat digunakan untuk melakukan prediksi banjir satu minggu ke depan.

Itu biasanya sudah sesuai dengan yang dimaksud dosen ketika menyebut "dt-1 sampai dt-6" dan "tentukan dulu rentang yang ingin diramalkan".

Kalau kamu mau, saya bisa langsung memodifikasi notebook .ipynb yang kamu upload dan membuatkan kode transformasi lag feature + target 7 hari ke depan yang siap ditempel ke notebook AutoML-mu.

Secara akademik, ada perbedaan yang cukup penting.

Peramalan (Forecasting)

Peramalan digunakan ketika target yang diprediksi berupa nilai yang akan datang berdasarkan urutan waktu (time series).

Contoh:

Curah hujan 7 hari ke depan
Tinggi muka air sungai besok
Jumlah penumpang bulan depan
Harga saham minggu depan

Output:

Tanggal 20 Juni : 25 mm
Tanggal 21 Juni : 18 mm
Tanggal 22 Juni : 30 mm

atau

Tinggi muka air = 1.8 meter

Ini forecasting/regresi time series.

Prediksi (Prediction / Classification)

Digunakan ketika target berupa kelas atau kategori.

Contoh:

Banjir atau tidak banjir
Macet atau tidak macet
Fraud atau tidak fraud
Lulus atau tidak lulus

Output:

Flood = 1

atau

Flood = 0

Ini klasifikasi.

Kasus Datasetmu

Target yang kamu gunakan adalah:

Flood
0 = Tidak Banjir
1 = Banjir

Artinya modelmu sebenarnya adalah:

Prediksi klasifikasi banjir.

Bukan forecasting murni.

Kenapa dosen menyebut time series?

Karena fitur yang digunakan berasal dari data berurutan berdasarkan waktu.

Misalnya:

Rain_t-1
Rain_t-2
Rain_t-3
...
Rain_t-7

Artinya:

Input = time series
Model = klasifikasi

Dalam literatur ini biasanya disebut:

Time Series Classification

atau

Flood Forecasting Classification

Kalau judulmu:

Prediksi Banjir Menggunakan AutoML

masih benar.

Kalau dosen meminta:

dibuat time series

yang dimaksud biasanya bukan mengubah penelitian menjadi forecasting curah hujan atau ARIMA, tetapi:

Data harus disusun secara temporal.
Gunakan lag feature (t-1, t-2, dst.).
Hindari data leakage dari masa depan.
Prediksi kejadian banjir pada waktu mendatang.

Jadi untuk penelitianmu:

Jenis data → Time Series
Bentuk target → Klasifikasi
Tugas machine learning → Binary Classification
Pendekatan → Time Series Classification / Flood Forecasting

Kalau ditanya penguji:

"Ini forecasting atau klasifikasi?"

Jawaban yang paling tepat:

Penelitian ini merupakan klasifikasi biner untuk memprediksi kejadian banjir di masa mendatang dengan memanfaatkan data time series cuaca yang ditransformasikan menjadi lag features. Jadi data yang digunakan bersifat time series, tetapi tugas machine learning yang dilakukan adalah klasifikasi, bukan forecasting nilai kontinu.

nah kalau memang ini prediksi, ngolah datasetnya yang sesuai itu gimana? apalagi dosenku ada ngomong tentang timeseries gitu

Menurut saya ada kemungkinan terjadi miskomunikasi antara kamu dan dosen.

Kamu berpikir:

"Kalau time series berarti harus forecasting seperti ARIMA, LSTM, meramalkan curah hujan."

Padahal dosen kemungkinan besar maksudnya:

"Data prediksi banjirmu harus dibentuk sebagai time series."

Itu dua hal yang berbeda.

Untuk kasusmu, saya justru tidak menyarankan mengubah penelitian menjadi forecasting murni, karena targetmu adalah Flood (0/1).

Yang perlu diubah adalah feature engineering.

Alur yang saya kira diinginkan dosen:

Dataset awal:

tanggal	hujan	suhu	soil	Flood
1 Jan	10	30	0.30	0
2 Jan	20	31	0.35	0
3 Jan	50	29	0.45	1

Saat ini modelmu kemungkinan melihat:

Hujan hari ini
Suhu hari ini
Soil hari ini

→ Flood hari ini

Ini bukan time-series learning.

Yang diminta dosen kemungkinan:

Hujan 7 hari terakhir
Suhu 7 hari terakhir
Soil moisture 7 hari terakhir

→ Flood besok

atau

→ Flood 3 hari lagi

atau

→ Flood 7 hari lagi

Visualisasinya:

Hari ke-10

Input:

Rain_t-7
Rain_t-6
Rain_t-5
Rain_t-4
Rain_t-3
Rain_t-2
Rain_t-1

Output:

Flood_t

atau

Output:

Flood_t+1

Saya justru menyarankan jangan langsung pakai horizon 7 hari.

Karena secara hidrologi:

Curah hujan hari ini
Soil moisture hari ini

lebih berpengaruh terhadap banjir:

besok
lusa
3 hari ke depan

daripada 7 hari lagi.

Kalau banjir 7 hari lagi, hubungan sebab-akibatnya mulai melemah.

Kalau saya yang menyusun penelitianmu, saya akan membuat:

Target:

Flood_t+1

Feature:

rain_lag1
rain_lag2
rain_lag3
rain_lag4
rain_lag5
rain_lag6
rain_lag7

soil_lag1
soil_lag2
...
soil_lag7

temp_max_lag1
...
temp_max_lag7

Bahkan lebih bagus lagi ditambah fitur agregasi.

Karena banjir biasanya dipengaruhi akumulasi hujan.

Contoh:

rain_3day_sum
rain_7day_sum
rain_14day_sum

Misalnya:

7 hari terakhir hujan

20
15
30
50
40
10
25

maka:

rain_7day_sum = 190 mm

Ini sering jauh lebih informatif daripada hanya lag mentah.

Kalau saya gambar pipeline yang paling masuk akal:

Data Cuaca Harian
        ↓
Sort berdasarkan waktu
        ↓
Buat lag feature
(t-1 ... t-7)
        ↓
Buat rolling feature
(3 hari, 7 hari)
        ↓
Target = Flood_t+1
        ↓
Time-based split
        ↓
AutoML
        ↓
Evaluasi Recall/F1

Tentang imbalance:

Jangan lakukan SMOTE sebelum membuat lag feature.

Urutannya harus:

Dataset asli
      ↓
Buat lag feature
      ↓
Buat target t+1
      ↓
Train-test split berdasarkan waktu
      ↓
SMOTE hanya pada training set

Banyak mahasiswa salah karena melakukan SMOTE sebelum split sehingga terjadi data leakage.

Jadi jika saya menerjemahkan komentar dosen:

perlu dibuat time series

= ubah dataset menjadi lag feature.

data perlu ditransformasi

= buat rain_t-1 sampai rain_t-6, soil_t-1 sampai soil_t-6, dst.

tentukan berapa lama diramalkan

= tentukan targetnya Flood_t+1, Flood_t+3, atau Flood_t+7.

Untuk datasetmu yang sangat imbalance (sekitar 56 kejadian banjir dari >6000 data), saya akan memilih:

Window observasi: 7 hari
Target: Flood_t+1
Tambahan: rolling rainfall 3 hari dan 7 hari
Split: berdasarkan waktu
Penanganan imbalance: class weight atau SMOTE setelah split

Ini yang paling mudah dipertanggungjawabkan saat sidang dibanding langsung memaksakan Flood_t+7.

jadi dataset yang tadi ku punya itu udah gimana dan harus gimana biar sesuai?

Kalau melihat struktur dataset_utama.csv yang kamu bangun dari script build_primary_dataset.py, datasetmu saat ini kira-kira berbentuk seperti ini:

city	time	precipitation	temp_max	temp_min	wind	rain	soil_moisture	Flood
Samarinda	2020-01-01	10	32	25	15	8	0.31	0
Samarinda	2020-01-02	15	31	24	14	12	0.33	0
Samarinda	2020-01-03	45	30	24	12	40	0.40	1

Ini disebut snapshot dataset atau tabular biasa.

Masalahnya:

Model melihat:

Cuaca tanggal 3 Januari
↓
Banjir tanggal 3 Januari

Padahal dosen ingin ada unsur temporal/time-series.

Datasetmu sekarang sebenarnya sudah benar sebagai dataset dasar, tetapi belum benar sebagai dataset time series untuk prediksi banjir.

Yang perlu diubah bukan mengumpulkan data baru.

Yang perlu diubah adalah bentuk record-nya.

Misalnya kita pilih:

Window observasi = 7 hari
Prediksi = 1 hari ke depan

Maka satu baris data berubah menjadi:

rain_t-7	rain_t-6	rain_t-5	rain_t-4	rain_t-3	rain_t-2	rain_t-1	soil_t-7	...	Flood_t+1
10	15	30	5	20	45	60	0.30	...	1

Interpretasinya:

Model melihat kondisi 7 hari terakhir

↓

Memprediksi apakah besok banjir

Ini baru time-series classification.

Kalau saya susun untuk penelitianmu:

Dataset Mentah (sekarang)
Tanggal	Rain	Soil	Flood
1 Jan	10	0.30	0
2 Jan	20	0.32	0
3 Jan	40	0.35	0
4 Jan	50	0.40	1
Dataset Setelah Transformasi
rain_lag7	rain_lag6	rain_lag5	rain_lag4	rain_lag3	rain_lag2	rain_lag1	Flood_target
10	20	40	50	25	30	60	1

Yang menurut saya paling cocok:

Feature lag

Untuk setiap variabel:

rain_sum
precipitation_sum
soil_moisture
temp_max
temp_min
wind_speed

buat:

lag1
lag2
lag3
lag4
lag5
lag6
lag7
Feature agregasi

Tambahkan:

rain_sum_3days
rain_sum_7days

precipitation_3days
precipitation_7days

soil_mean_3days
soil_mean_7days

Karena banjir lebih dipengaruhi akumulasi hujan daripada hujan satu hari.

Target

Saya sarankan:

Flood_t+1

bukan:

Flood_t+7

karena:

Jumlah kejadian banjirmu hanya sekitar 56.
Forecast 7 hari terlalu jauh untuk dataset cuaca harian.
Relasi sebab-akibat cuaca → banjir lebih kuat pada 1–3 hari.

Jadi kesimpulan praktisnya:

Dataset yang kamu punya sudah cukup. Tidak ada dataset wajib yang kurang.

Yang harus dilakukan adalah:

Urutkan data berdasarkan kota dan tanggal.
Buat lag feature (t-1 sampai t-7).
Buat rolling feature (3 hari dan 7 hari).
Buat target Flood_t+1.
Split train-test berdasarkan waktu.
Baru tangani imbalance (SMOTE/class weight) pada data training.

Kalau dosenmu benar-benar mengucapkan:

"dt-1, dt-2, sampai dt-6"

maka hampir pasti yang beliau maksud adalah transformasi lag feature seperti di atas, bukan mencari dataset tambahan atau mengganti model menjadi ARIMA/LSTM.