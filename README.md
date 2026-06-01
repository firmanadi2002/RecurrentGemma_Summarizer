# RecurrentGemma Summarizer 📝🇮🇩

Proyek Tugas Akhir ini berfokus pada pengembangan sistem peringkas teks otomatis (*text summarizer*) khusus untuk artikel berita berbahasa Indonesia. Sistem ini menggunakan model **RecurrentGemma** (arsitektur berbasis Griffin yang efisien dalam penggunaan memori) yang melalui proses *fine-tuning* dengan framework **JAX**.

Untuk mempermudah penggunaan, proyek ini juga dilengkapi dengan aplikasi antarmuka web interaktif berbasis **Streamlit**.

---

## 🚀 Fitur Utama
- **Peringkas Berita Otomatis:** Mampu mereduksi teks berita berbahasa Indonesia yang panjang menjadi ringkasan singkat tanpa kehilangan konteks utamanya.
- **Antarmuka Web Interaktif:** Dilengkapi dengan aplikasi Streamlit, sehingga pengguna cukup menempelkan teks berita ke browser untuk mendapatkan hasil ringkasan secara instan.

## 📊 Hasil & Spesifikasi
Proses *fine-tuning* yang dilakukan pada dataset berita bahasa Indonesia menghasilkan performa model yang optimal untuk Tugas Akhir ini:
- **Akurasi Model:** Berhasil mencapai skor evaluasi/akurasi sebesar **0.89**.
- **Ukuran Model (*Fine-tuned*):** ~5 GB.
- **Ukuran Dataset:** ~64 MB.

---

## 📂 Struktur Repositori
Karena batasan ukuran file di GitHub, file model hasil *fine-tuning* (5 GB) dan dataset (64 MB) tidak dimasukkan secara langsung ke dalam repositori ini. Keduanya disediakan melalui tautan Google Drive. Repositori ini fokus menampung berkas kode utama:

```text
├── data_training/
│   └── trainingModel.ipynb  # Notebook proses fine-tuning model menggunakan JAX
├── app.py                   # Program utama aplikasi web Streamlit
└── README.md                # Dokumentasi proyek
```

## Panduan Instalasi & Prasyarat
Untuk konfigurasi lingkungan, instalasi komponen utama JAX, serta dependensi dasar RecurrentGemma, proyek ini sepenuhnya merujuk pada repositori resmi Google.
1. Ikuti langkah-langkah instalasi lingkungan (environment) secara lengkap pada repositori asli: https://github.com/google-deepmind/recurrentgemma
2. Setelah lingkungan JAX dan RecurrentGemma siap, lakukan klon pada repositori ini
3. Unduh file model yang telah di-fine-tune serta dataset melalui tautan di bawah ini, lalu simpan ke dalam direktori proyek Anda: https://s.id/Dataset_Model
