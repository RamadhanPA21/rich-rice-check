# RICH (Rice Check) - Deteksi Penyakit Daun Padi

RICH adalah proyek Deep Learning berbasis Computer Vision yang dirancang untuk mengklasifikasikan jenis penyakit pada daun padi secara otomatis. Proyek ini menggunakan model Convolutional Neural Network (CNN) yang dilatih menggunakan framework Keras dan disajikan dalam bentuk antarmuka web interaktif menggunakan Streamlit.

## 📊 Performa Model & Analisis
Model dilatih menggunakan skrip `train.py` dan menyimpan bobot terbaiknya di `best_rice_disease_model.keras`. Anda dapat melihat detail performa evaluasi model pada file visual berikut di dalam repositori:
* **Grafik Akurasi:** Lihat perkembangan akurasi pada `accuracy_graph.png`
* **Grafik Loss:** Lihat minimalisasi error pada `loss_graph.png`
* **Confusion Matrix:** Lihat detail ketepatan prediksi tiap kelas penyakit pada `confusion_matrix.png`

## 💻 Panduan Instalasi & Cara Menjalankan
Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi RICH secara lokal di komputer Anda:

1. **Membuat Virtual Environment:**
   ```bash
   python -m venv .venv
   ```

2. **Mengaktifkan Virtual Environment (Windows):**
   ```bash
   .venv\Scripts\activate
   ```

3. **Menjalankan Aplikasi Utama (Streamlit):**
   ```bash
   streamlit run app.py
   ```
