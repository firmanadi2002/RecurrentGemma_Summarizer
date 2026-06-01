# RecurrentGemma Summarizer 📝🇮🇩

Proyek Tugas Akhir ini berfokus pada pengembangan sistem peringkas teks otomatis (*text summarizer*) khusus untuk artikel berita berbahasa Indonesia. Sistem ini menggunakan model **RecurrentGemma** (arsitektur berbasis Griffin yang efisien dalam penggunaan memori) yang melalui proses *fine-tuning* dengan framework **JAX**.

Untuk mempermudah penggunaan, proyek ini juga dilengkapi dengan aplikasi antarmuka web interaktif berbasis **Streamlit**.

---

## 🚀 Fitur Utama
- **Peringkas Berita Otomatis:** Mampu mereduksi teks berita berbahasa Indonesia yang panjang menjadi ringkasan singkat tanpa kehilangan konteks utamanya.
- **Antarmuka Web Interaktif:** Dilengkapi dengan aplikasi Streamlit, sehingga pengguna cukup menempelkan teks berita ke browser untuk mendapatkan hasil ringkasan secara instan.
- **Arsitektur Efisien:** Menggunakan keunggulan RecurrentGemma yang mengombinasikan *Linear Recurrent* dan *Local Attention* untuk pemrosesan yang cepat.

## 📊 Hasil & Performa Model
Proses *fine-tuning* yang dilakukan pada dataset berita bahasa Indonesia menghasilkan performa model yang optimal untuk Tugas Akhir ini:
- **Akurasi Model:** Berhasil mencapai skor evaluasi atau akurasi sebesar **0.89**.
- **Ukuran Model:** ~64 MB, menjadikannya jauh lebih ringan dan efisien untuk dijalankan secara lokal dibandingkan arsitektur Transformer standar.

---

## 📂 Struktur Repositori
Karena batasan ukuran file di GitHub, bobot model hasil training (~64MB) dan dataset tidak dimasukkan secara langsung ke dalam repositori ini, melainkan disediakan melalui tautan Google Drive. Repositori ini fokus menampung berkas kode utama:

```text
├── data_training/
│   └── trainingModel.ipynb  # Notebook proses fine-tuning model menggunakan JAX
├── app.py                   # Program utama aplikasi web Streamlit
└── README.md                # Dokumentasi proyek

