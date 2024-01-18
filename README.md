# clusteringdoc
========== Documents Clustering menggunakan KMEANS dan TF-IDF ===========



Dataset diperoleh dari kaggle dan diperkecil skalanya
source: https://www.kaggle.com/code/jbencina/clustering-documents-with-tfidf-and-kmeans


=> Permasalahan dan Tujuan Eksperimen
* Permasalahan:
Eksperimen ini berfokus pada pengelompokan (klustering) dokumen teks secara otomatis. Dalam banyak kasus, dokumen teks (seperti artikel, laporan, atau dokumen lain) seringkali tidak dilabeli atau dikategorikan, yang membuat pengelolaan dan pencarian informasi menjadi lebih sulit. Klustering dokumen membantu mengatasi masalah ini dengan mengelompokkan dokumen berdasarkan kemiripan konten.

* Tujuan:
Tujuan utama dari eksperimen ini adalah untuk mengelompokkan dokumen teks ke dalam cluster atau kelompok yang berbeda berdasarkan konten atau topiknya. Hal ini dimaksudkan untuk memudahkan akses dan manajemen dokumen. Eksperimen ini juga bertujuan untuk mengevaluasi efektivitas model klustering dalam mengidentifikasi dan memisahkan topik yang berbeda dalam kumpulan dokumen.

=> Model dan Alur Tahapan Eksperimen

Model:
- TF-IDF (Term Frequency-Inverse Document Frequency): Metode ini digunakan untuk mengkonversi teks dokumen menjadi format vektor numerik. TF-IDF mengukur pentingnya suatu kata dalam dokumen dalam kumpulan data.
- KMeans Clustering: Algoritma klustering yang digunakan untuk mengelompokkan dokumen ke dalam cluster berdasarkan fitur vektor TF-IDF.

=> Alur Tahapan Eksperimen:
- Pra-pemrosesan Data: Membersihkan teks dan menyiapkan data (seperti menghapus tanda baca, melakukan tokenisasi, dan menghilangkan stop words).
- Penerapan TF-IDF: Mengubah dokumen teks menjadi matriks fitur TF-IDF.
- Klustering dengan KMeans: Menerapkan algoritma KMeans pada matriks TF-IDF untuk membentuk cluster.
- Evaluasi Model: Menentukan jumlah cluster yang optimal dan mengevaluasi kualitas klustering.
- Performa Model / Uji Performa Model


Untuk mengevaluasi performa model, dapat digunakan beberapa metrik, seperti:

- Silhouette Score: Mengukur seberapa baik dokumen dikelompokkan. Nilai tinggi menunjukkan bahwa dokumen cocok dengan baik di dalam cluster mereka dan tidak cocok dengan cluster lain.
- Elbow Method: Menentukan jumlah cluster optimal dengan menganalisis perubahan nilai SSE (Sum of Squared Errors).

=> Proses Deployment

Pengembangan aplikasi web menggunakan Streamlit untuk menampilkan hasil klustering dan memungkinkan pengguna untuk berinteraksi dengan model.

Langkah Deployment:
===> Menggunakan Streamlit Sharing
Streamlit Sharing adalah platform hosting gratis dari Streamlit yang dirancang khusus untuk aplikasi Streamlit.

1. GitHub Repository: Pastikan kode aplikasi Streamlit Anda tersimpan di GitHub. Repositori harus mencakup file requirements.txt dengan semua dependensi yang diperlukan.

2. Daftar untuk Streamlit Sharing: Kunjungi Streamlit Sharing dan daftar untuk mendapatkan akses.

3. Deploy dari GitHub: Setelah mendapatkan akses, melakukan deploy aplikasi langsung dari GitHub ke Streamlit Sharing. disini perlu memberikan tautan ke repositori GitHub Anda.

4. Konfigurasi dan Launch: Konfigurasikan opsi yang diperlukan dan luncurkan aplikasi. Streamlit Sharing akan menangani instalasi dependensi dan menjalankan aplikasi.
