# 🎮 Simulasi Pertumbuhan Pengguna Global Gaming & eSports

Aplikasi web Streamlit untuk simulasi pertumbuhan pengguna gaming dan eSports menggunakan model Logistic Growth yang diselesaikan dengan metode Runge-Kutta Orde 4 (RK4).

## 🚀 Deployment ke Railway

### Prerequisites
- Akun Railway (https://railway.app)
- Git repository (GitHub/GitLab/Bitbucket)

### Langkah-langkah Deployment

1. **Push code ke Git Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy ke Railway**
   - Login ke [Railway](https://railway.app)
   - Klik "New Project"
   - Pilih "Deploy from GitHub repo" (atau GitLab/Bitbucket)
   - Pilih repository Anda
   - Railway akan otomatis mendeteksi `requirements.txt` dan `Procfile`

3. **Konfigurasi Environment Variables** (Opsional)
   - Tidak diperlukan untuk aplikasi ini
   - Jika perlu, bisa ditambahkan di Railway dashboard

4. **Deploy**
   - Railway akan otomatis build dan deploy aplikasi
   - Tunggu hingga build selesai
   - Aplikasi akan tersedia di URL yang diberikan Railway

### File-file Penting untuk Deployment

- `requirements.txt` - Dependencies Python
- `Procfile` - Command untuk menjalankan aplikasi
- `railway.json` - Konfigurasi Railway (opsional)
- `.streamlit/config.toml` - Konfigurasi Streamlit
- `app.py` - Aplikasi utama

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run aplikasi
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## 📋 Fitur

- 📊 Visualisasi data gaming & eSports
- 🔢 Simulasi Logistic Growth dengan RK4
- 📈 Interactive plots dengan Plotly
- 📥 Download hasil simulasi (CSV)
- ⚙️ Parameter tuning melalui sidebar

## 🛠️ Tech Stack

- **Streamlit** - Web framework
- **Pandas** - Data processing
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations

## 📝 Catatan

- Aplikasi memerlukan upload file CSV dataset
- Dataset harus memiliki kolom "Year" dan variabel numerik lainnya
- Model menggunakan Logistic Growth: `dU/dt = r*U*(1 - U/K)`
