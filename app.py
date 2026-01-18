import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Logistic Growth Simulator (RK4)",
    layout="centered"
)

st.title("📈 Logistic Growth Simulator (RK4)")
st.write(
    "Simulasi pertumbuhan pengguna platform digital "
    "menggunakan **Model Logistic Growth** "
    "dan **Metode Runge–Kutta Orde 4 (RK4)**."
)

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("global_gaming_esports_2010_2025.csv")

df = load_data()

# =========================
# PILIH DATA
# =========================
st.subheader("📊 Dataset Global Gaming & eSports Growth")

st.dataframe(df.head())

t_data = df["Year"].values
U_data = df["Player_Count"].values

# Normalisasi waktu
t = t_data - t_data[0]

# =========================
# MODEL LOGISTIC
# =========================
def logistic_growth(U, r, K):
    return r * U * (1 - U / K)

# =========================
# RK4 IMPLEMENTATION
# =========================
def rk4_logistic(U0, t, r, K):
    U = np.zeros(len(t))
    U[0] = U0
    
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        
        k1 = logistic_growth(U[i - 1], r, K)
        k2 = logistic_growth(U[i - 1] + 0.5 * h * k1, r, K)
        k3 = logistic_growth(U[i - 1] + 0.5 * h * k2, r, K)
        k4 = logistic_growth(U[i - 1] + h * k3, r, K)
        
        U[i] = U[i - 1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return U

# =========================
# SIDEBAR PARAMETER
# =========================
st.sidebar.header("⚙️ Parameter Model")

U0 = st.sidebar.slider(
    "Jumlah Pengguna Awal U(0)",
    min_value=float(min(U_data)),
    max_value=float(max(U_data)),
    value=float(U_data[0])
)

K = st.sidebar.slider(
    "Kapasitas Pasar (K)",
    min_value=float(max(U_data)),
    max_value=float(max(U_data) * 2),
    value=float(max(U_data) * 1.2)
)

r = st.sidebar.slider(
    "Laju Pertumbuhan (r)",
    min_value=0.01,
    max_value=1.0,
    value=0.25,
    step=0.01
)

# =========================
# SIMULASI
# =========================
U_sim = rk4_logistic(U0, t, r, K)

# =========================
# VISUALISASI
# =========================
st.subheader("📉 Hasil Simulasi")

fig, ax = plt.subplots()
ax.plot(t_data, U_data, label="Data Riil")
ax.plot(t_data, U_sim, linestyle="--", label="Simulasi Logistic (RK4)")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Players")
ax.set_title("Logistic Growth Simulation using RK4")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =========================
# EVALUASI ERROR
# =========================
mse = np.mean((U_data - U_sim) ** 2)

st.subheader("📐 Evaluasi Model")
st.write(f"**Mean Squared Error (MSE):** {mse:.4e}")

# =========================
# INTERPRETASI
# =========================
st.subheader("📝 Interpretasi Singkat")
st.write(
    """
    - Model Logistic Growth mampu merepresentasikan pola pertumbuhan pengguna.
    - Metode RK4 memberikan solusi numerik yang stabil dan akurat.
    - Perbedaan antara data riil dan simulasi dapat dikurangi dengan penyesuaian parameter **r** dan **K**.
    """
)
