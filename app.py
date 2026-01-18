import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Simulasi Pertumbuhan Pengguna Gaming & eSports",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Simulasi Pertumbuhan Pengguna Global Gaming & eSports")
st.markdown("""
Model **Logistic Growth (kontinu)** diselesaikan dengan  
**Metode Runge–Kutta Orde 4 (RK4)**.
""")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.sort_values("Year")
    return df

uploaded_file = st.file_uploader(
    "Upload dataset Global Gaming & eSports (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file `global_gaming_esports_2010_2025.csv`")
    st.stop()

df = load_data(uploaded_file)

# ===============================
# DATA PREVIEW
# ===============================
with st.expander("📋 Data Preview"):
    st.dataframe(df, use_container_width=True)

# ===============================
# SELECT TARGET
# ===============================
target_col = st.selectbox(
    "Pilih Variabel Pertumbuhan",
    [
        "Active_Players_Million", 
        "Esports_Viewers_Million", 
        "Gaming_Revenue_BillionUSD", 
        "Esports_Revenue_MillionUSD",
        "Avg_Spending_USD",
        "Esports_Tournaments_Count",
        "Pro_Players_Count",
        "Internet_Penetration_Percent",
        "Avg_Latency_ms",
        "AR_VR_Adoption_Index",
        "Streaming_Influence_Index",
        "Covid_Impact_Index",
        "Female_Gamer_Percent",
        "Mobile_Gaming_Share",
        "Esports_PrizePool_MillionUSD",
        "Gaming_Companies_Count"
    ]
)

t_data = df["Year"].values
U_data = df[target_col].values
t = t_data - t_data[0]

# ===============================
# LOGISTIC MODEL
# ===============================
def logistic_growth(U, r, K):
    return r * U * (1 - U / K)

def rk4(U0, t, r, K):
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

# ===============================
# SIDEBAR PARAMETERS
# ===============================
st.sidebar.header("⚙️ Parameter Model")

r = st.sidebar.slider(
    "Laju Pertumbuhan (r)",
    0.01, 1.0, 0.25, 0.01
)

K = st.sidebar.slider(
    "Kapasitas Maksimum (K)",
    float(U_data.max()),
    float(U_data.max() * 3),
    float(U_data.max() * 1.2),
    step=1e6
)

# ===============================
# RUN SIMULATION
# ===============================
U0 = U_data[0]
U_sim = rk4(U0, t, r, K)

# ===============================
# METRICS
# ===============================
rmse = np.sqrt(np.mean((U_data - U_sim)**2))
mae = np.mean(np.abs(U_data - U_sim))

# ===============================
# KPI DISPLAY
# ===============================
col1, col2, col3 = st.columns(3)

col1.metric("Initial Value", f"{U0:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("MAE", f"{mae:,.2f}")

# ===============================
# VISUALIZATION
# ===============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t_data,
    y=U_data,
    mode="markers+lines",
    name="Data Riil",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    x=t_data,
    y=U_sim,
    mode="lines",
    name="Simulasi Logistic (RK4)",
    line=dict(width=4, dash="dash")
))

fig.update_layout(
    title="Pertumbuhan Varibel Terpilih Global Gaming & eSports",
    xaxis_title="Tahun",
    yaxis_title=target_col,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# MODEL FORMULATION
# ===============================
with st.expander("📐 Model Matematis"):
    st.latex(r"\frac{dU(t)}{dt} = rU(t)\left(1 - \frac{U(t)}{K}\right)")
    st.markdown("""
**Keterangan:**
- \(U(t)\): nilai variabel pada waktu ke-\(t\)
- \(r\): laju pertumbuhan intrinsik
- \(K\): kapasitas maksimum/teoritis
""")

# ===============================
# EXPORT
# ===============================
export_df = df.copy()
export_df["Simulation_RK4"] = U_sim

csv_data = export_df.to_csv(index=False)

st.download_button(
    "⬇️ Download Hasil Simulasi (CSV)",
    csv_data,
    file_name="hasil_simulasi_logistic_rk4.csv",
    mime="text/csv"
)
