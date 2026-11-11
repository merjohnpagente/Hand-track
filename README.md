# Retrolens - Hand Tracking Filter & Portal

Aplikasi filter kamera interaktif menggunakan gesture tangan (hand tracking) dengan MediaPipe dan OpenCV. Anda dapat membuat "portal" dengan jari Anda yang akan menerapkan berbagai filter menarik (Mono, Dual-Tone, Pixelate, Invert, Sepia, Blur, Thermal, Sketch, Glitch, Neon, Galaxy).

## Versi Web (GitHub Pages)

Repo ini juga memiliki versi web yang berjalan sepenuhnya di browser menggunakan MediaPipe Tasks Vision (WebAssembly). Tidak perlu install Python atau library apa pun.

- **Coba online:** https://merjohnpagente.github.io/Hand-track/
- **Jalankan lokal:** dari root repo jalankan `python3 -m http.server 8000`, lalu buka `http://localhost:8000/web/` (butuh internet untuk memuat MediaPipe dari CDN).
- **Deploy:** setiap push ke branch `main` otomatis di-build dan dipublikasikan oleh GitHub Actions (lihat `.github/workflows/deploy-pages.yml`). Aktifkan sekali di **Settings → Pages → Source: GitHub Actions** jika belum aktif.
- **Deploy ke Vercel:** import repo ini di https://vercel.com/new. Vercel otomatis memakai `vercel.json` (build command `bash scripts/build-site.sh`, output `dist/`), jadi cukup klik **Deploy**. Setiap push ke `main` juga akan redeploy otomatis.

Semua pemrosesan video terjadi di perangkat pengguna; tidak ada frame yang dikirim ke server.

## Persyaratan Sistem
- Python 3.7 atau lebih baru
- Webcam

## Cara Install dan Menjalankan

### 1. Clone Repository
Pertama, clone repository ini ke komputer Anda dan masuk ke foldernya (ganti URL dengan link repository GitHub Anda):
```bash
git clone <URL_GITHUB_ANDA>
cd <NAMA_FOLDER_REPO>
```

### 2. Buat Virtual Environment (Opsional tapi Sangat Disarankan)
Gunakan virtual environment agar dependencies (library) tidak bentrok dengan project Python lainnya di komputer Anda.
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install semua library Python yang dibutuhkan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```

Atau jika Anda ingin menginstallnya secara manual satu per satu:
```bash
pip install opencv-python mediapipe numpy
```

### 4. Pastikan Model MediaPipe Tersedia
Aplikasi ini membutuhkan dua file model dari MediaPipe yang seharusnya sudah ada di dalam repository ini:
1. `hand_landmarker.task` (Untuk mendeteksi titik pada tangan)
2. `selfie_segmenter.tflite` (Untuk filter Galaxy / memisahkan background)

Jika file tersebut belum ada, pastikan untuk meletakkannya di dalam folder yang sama dengan file `main.py`.

### 5. Jalankan Aplikasi
Jalankan script utamanya dengan mengetik:
```bash
python main.py
```
*(Gunakan `python3 main.py` jika Anda menggunakan macOS/Linux dan tidak menggunakan virtual environment)*

## Cara Penggunaan Fitur

- **Membuka Portal:** Gunakan ujung telunjuk dan jempol dari **kedua** tangan Anda di depan kamera (total 4 jari). Sebuah portal berbentuk persegi empat akan terbentuk di antara keempat jari Anda, dan efek filter akan muncul di dalamnya.
- **Mengganti Filter:** Ada beberapa cara untuk mengganti filter yang sedang aktif:
  - Sentuhkan/dekatkan ujung jempol dan jari kelingking Anda.
  - Atau, dekatkan ujung telunjuk dari kedua tangan Anda.
- **Menutup Aplikasi:** Pastikan jendela kamera/Retrolens sedang aktif (diklik), kemudian tekan tombol **`q`** pada keyboard Anda untuk keluar dari aplikasi.

---
**Catatan:** Pastikan ruangan memiliki pencahayaan yang cukup agar deteksi tangan dari kamera dapat bekerja dengan optimal.
  