import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset_stunting.csv")

# Pisahkan fitur dan target
X = df.drop(columns=["Status Stunting"])
y = df["Status Stunting"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load Model
rf_model = joblib.load("random_forest_stunting.pkl")

# Styling UI
st.set_page_config(page_title="Prediksi Stunting", layout="wide")
st.markdown("""
    <style>
        .sidebar .sidebar-content { background-color: #1E1E1E; color: white; }
        .css-1d391kg { padding-top: 1rem; }
        h1 { color: #FFA500; }
        .stButton > button { background-color: #FFA500; color: white; border-radius: 5px; }
        .stButton > button:hover { background-color: #FF8C00; }
    </style>
""", unsafe_allow_html=True)

def update_menu():
    st.session_state["menu"] = st.session_state["menu_selector"]

# Sidebar Navigasi
st.sidebar.image("logo_stunting.png", width=200)
st.sidebar.title("🔍 Navigasi")
st.sidebar.markdown("---")

# Radio Button dengan on_change agar perubahan langsung dieksekusi
menu = st.sidebar.radio(
    "Pilih Halaman:", 
    ["Home", "Prediksi", "Visualisasi", "Algoritma"], 
    index=["Home", "Prediksi", "Visualisasi", "Algoritma"].index(st.session_state.get("menu", "Home")),
    key="menu_selector",
    on_change=update_menu  # Perbarui menu secara otomatis
)

st.sidebar.markdown("---")
st.sidebar.info("Gunakan aplikasi ini untuk memahami lebih dalam tentang stunting dan bagaimana cara mencegahnya.")

# Update session_state sesuai pilihan menu
st.session_state["menu"] = menu

# --- MENU HOME ---
if menu == "Home":

    st.title("📊 Sistem Prediksi Stunting pada Balita")
    st.markdown(
        """
        **Stunting** adalah kondisi gagal tumbuh pada balita akibat kekurangan gizi kronis, infeksi berulang, 
        dan stimulasi psikososial yang kurang memadai. Kondisi ini berdampak **jangka panjang**, termasuk gangguan perkembangan otak 
        dan peningkatan risiko penyakit di masa dewasa.
        """, unsafe_allow_html=True
    )

    # Layout dengan 2 kolom
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Tujuan Aplikasi")
        st.write("- 📌 **Memprediksi** status stunting berdasarkan data balita.")
        st.write("- 📊 **Memberikan insight** dari data stunting dengan visualisasi interaktif.")
        st.write("- 🏆 **Menjelaskan pemilihan algoritma terbaik** untuk klasifikasi stunting.")
        st.write("- 🩺 **Membantu tenaga kesehatan dan orang tua** dalam pemantauan pertumbuhan anak.")

        # Coba Prediksi Sekarang
        st.subheader("🔍 Coba Prediksi Sekarang!")
        st.markdown("Klik tombol di bawah untuk mulai menggunakan sistem prediksi.")        
        if st.button("🟠 Mulai Prediksi", on_click=lambda: st.session_state.update(menu="Prediksi")):
            st.rerun()


    with col2:
        st.subheader("📈 Fakta Stunting")
        st.info("🔹 **1 dari 4 anak di Indonesia mengalami stunting.**")
        st.info("🔹 Kekurangan gizi dalam **1000 hari pertama kehidupan** sangat berpengaruh.")
        st.info("🔹 Stunting dapat dicegah dengan pola makan sehat dan perawatan yang baik.")

# --- MENU PREDIKSI ---
if menu == "Prediksi":
    st.title("🔍 Prediksi Stunting pada Balita")
    st.write("Masukkan data balita di bawah ini untuk mendapatkan prediksi status stunting.")

    # Buat form agar lebih rapi
    with st.form("prediksi_form"):
        col1, col2 = st.columns(2)  # Bagi layout jadi dua kolom

        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            usia = st.number_input("Usia (bulan)", min_value=0, max_value=60, step=1)
            lila = st.number_input("Lingkar Lengan Atas (cm)", min_value=5.0, max_value=30.0, step=0.1)

        with col2:
            berat = st.number_input("Berat Badan (kg)", min_value=0.0, max_value=30.0, step=0.1)
            tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=120.0, step=0.1)

        # Encode Jenis Kelamin
        jk_encoded = 1 if jenis_kelamin == "Perempuan" else 0
        input_data = np.array([[jk_encoded, usia, berat, tinggi, lila]])

        # Tombol prediksi di dalam form
        submit = st.form_submit_button("🔍 Prediksi")

        if submit:
            # Cek input sebelum prediksi
            st.write(f"Jenis Kelamin Encoded: {jk_encoded}")
            st.write(f"Input Data ke Model: {input_data}")

            # Dapatkan probabilitas prediksi dari model
            with st.spinner("Model sedang memproses..."):
                y_probs = rf_model.predict_proba(input_data)
                st.write(f"Probabilitas Prediksi: {y_probs}")

            # Mengubah threshold prediksi dari 0.5 ke 0.6
            if y_probs[0][1] >= 0.6:  # Jika probabilitas Stunting >= 60%
                prediksi = [1]  # Stunting
            else:
                prediksi = [0]  # Tidak Stunting
            
            # Simpan hasil prediksi di session_state
            st.session_state["hasil_prediksi"] = "STUNTING" if prediksi[0] == 1 else "NORMAL"
            st.session_state["confidence"] = y_probs[0][prediksi[0]]  # Simpan confidence level
            
            # Tetap di halaman Prediksi
            st.session_state["menu"] = "Prediksi"
            st.rerun()

        # Tampilkan hasil jika ada
        if "hasil_prediksi" in st.session_state:
            confidence = st.session_state["confidence"] * 100  # Konversi ke persen
            if st.session_state["hasil_prediksi"] == "STUNTING":
                st.error(f"⚠️ **Hasil Prediksi: Balita mengalami STUNTING**\n🧠 **Confidence Level: {confidence:.2f}%**")
                st.warning(
                    "🩺 **Saran:** Segera konsultasikan dengan dokter atau tenaga kesehatan.\n\n"
                    "💡 Pastikan balita mendapatkan asupan gizi yang cukup dan pemeriksaan kesehatan berkala.\n\n"
                    "🍎 Nutrisi yang baik seperti protein, zat besi, dan vitamin sangat penting dalam 1000 hari pertama kehidupan balita."
                )
            else:
                st.success(f"✅ **Hasil Prediksi: Balita dalam kondisi NORMAL**\n🧠 **Confidence Level: {confidence:.2f}%**")
                st.info(
                    "🎉 **Balita dalam kondisi sehat!**\n\n"
                    "🥗 Tetap berikan makanan bergizi seimbang.\n\n"
                    "⚖️ Pantau pertumbuhan secara berkala untuk memastikan perkembangan optimal."
                )

                # --- Penjelasan Confidence Level ---
            st.info(
                "🧠 **Apa itu Confidence Level?**\n"
                "Confidence Level menunjukkan seberapa yakin model terhadap prediksi yang diberikan. Semakin tinggi persentasenya, semakin besar keyakinan model bahwa hasilnya benar.\n\n"
                "📌 **Interpretasi Confidence Level:**\n"
                "- **80% - 100%** → Model sangat yakin dengan prediksi.\n"
                "- **60% - 79%** → Model cukup yakin, namun tetap disarankan cek lebih lanjut.\n"
                "- **<60%** → Model tidak terlalu yakin, disarankan konsultasi dengan dokter untuk kepastian lebih lanjut.\n\n"
                "🩺 **Jika ragu dengan hasil prediksi, selalu konsultasikan dengan tenaga kesehatan.**"
            )

# --- MENU VISUALISASI ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_stunting.csv")
    
    # Ubah nilai numerik menjadi label
    df["JK"] = df["JK"].map({0: "Laki-laki", 1: "Perempuan"})
    df["Status Stunting"] = df["Status Stunting"].map({0: "Tidak Stunting", 1: "Stunting"})
    
    return df

df = load_data()

# Menu Visualisasi
if menu == "Visualisasi":
    # --- DASHBOARD ---
    st.title("📊 Eksploratory Data Analysis")

    # Buat dua kolom
    col1, col2 = st.columns([1.2, 1])  # Proporsi lebar: col1 lebih besar dari col2

    # **Kolom Kiri - Penjelasan Dataset**
    with col1:
        st.markdown("## 📜 Tentang Dataset")
        st.write("""
        Dataset ini berasal dari **data.go.id** dengan jumlah data **1.288 balita**.  
        Dataset ini berisi informasi mengenai **Antropometri pada balita di Indonesia** dan mencakup beberapa variabel penting seperti:
        - **Jenis Kelamin** (Laki-laki / Perempuan)
        - **Usia** (dalam bulan)
        - **Berat Badan** (dalam kg)
        - **Tinggi Badan** (dalam cm)
        - **Lingkar Lengan Atas (LiLA)** (dalam cm)  

        **Tujuan Penggunaan:**  
        Dataset ini digunakan sebagai **data historis** untuk membangun model prediksi **Stunting**, sehingga dapat membantu dalam pengambilan keputusan terkait **pencegahan dan penanganan stunting pada balita**.
        """)

    # **Kolom Kanan - Tabel Dataset**
    with col2:
        st.markdown("## 🔍 Tinjauan Dataset")
        st.dataframe(df.head(10))  # Tampilkan 10 data pertama

    # Visualisasi Data Stunting
    st.markdown("## 📊 Visualisasi Data Stunting")
    st.write("Berikut adalah visualisasi interaktif mengenai distribusi dan hubungan variabel dalam dataset.")

    # Distribusi Jenis Kelamin
    st.subheader("Distribusi Jenis Kelamin")
    gender_counts = df["JK"].value_counts()

    fig1 = px.pie(names=gender_counts.index, values=gender_counts.values, hole=0.3)
    st.plotly_chart(fig1)

    total_balita = df.shape[0]
    st.info(f"**Insight**: Dataset terdiri dari {total_balita} balita, dengan {gender_counts['Laki-laki']} ({gender_counts['Laki-laki'] / total_balita * 100:.2f}%) laki-laki dan {gender_counts['Perempuan']} ({gender_counts['Perempuan'] / total_balita * 100:.2f}%) perempuan.")

    # Distribusi Status Stunting
    st.subheader("Distribusi Status Stunting")
    st.bar_chart(df["Status Stunting"].value_counts())

    status_counts = df["Status Stunting"].value_counts()
    st.info(f"**Insight**: {status_counts['Stunting']} balita ({status_counts['Stunting'] / total_balita * 100:.2f}%) mengalami stunting, sedangkan {status_counts['Tidak Stunting']} balita ({status_counts['Tidak Stunting'] / total_balita * 100:.2f}%) tidak mengalami stunting.")

    # Scatter Plot Berat vs Tinggi
    st.subheader("Hubungan Berat vs Tinggi")
    fig2 = px.scatter(df.dropna(), x="Berat", y="Tinggi", color="Status Stunting", opacity=0.7)
    st.plotly_chart(fig2)

    st.info("**Insight**: Balita dengan stunting cenderung memiliki tinggi lebih rendah untuk berat yang sama dibandingkan balita yang tidak stunting.")
    st.info("**Insight**: Balita tidak stunting lebih tersebar, sedangkan balita stunting lebih terkonsentrasi di berat rendah.")

    # Korelasi Variabel Numerikal
    st.subheader("Korelasi Variabel")

    # Memastikan hanya variabel numerik yang dihitung korelasinya
    corr = df.select_dtypes(include=['number']).corr()

    # Plot heatmap korelasi
    fig3, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig3)

    # Insight Otomatis Berdasarkan Korelasi
    if 'Tinggi' in corr.columns and 'Berat' in corr.columns:
        tinggi_berat_corr = corr.loc['Tinggi', 'Berat']
        st.info(f"**Insight**: Tinggi dan Berat memiliki korelasi sangat tinggi ({tinggi_berat_corr:.2f}), "
                "menunjukkan bahwa semakin tinggi balita, semakin besar beratnya.")

    if 'Usia' in corr.columns:
        usia_tinggi_corr = corr.loc['Usia', 'Tinggi'] if 'Tinggi' in corr.columns else None
        usia_berat_corr = corr.loc['Usia', 'Berat'] if 'Berat' in corr.columns else None
        st.info(f"**Insight**: Usia berkorelasi kuat dengan Tinggi ({usia_tinggi_corr:.2f}) dan Berat ({usia_berat_corr:.2f}), "
                "yang logis karena semakin bertambah usia, anak bertambah tinggi dan berat.")

    if 'LiLA' in corr.columns:
        lila_corr = corr.loc['LiLA'].drop('LiLA').abs().max()  # Maks korelasi LiLA dengan variabel lain
        st.info(f"**Insight**: LiLA memiliki korelasi yang rendah dengan variabel lain (maks {lila_corr:.2f}), "
                "mengindikasikan fitur ini mungkin kurang signifikan dalam prediksi stunting.")

# --- MENU ALGORITMA ---
if menu == "Algoritma":
    st.title("🔍 Pemilihan Algoritma: **Random Forest**")
    
    # Layout 2 kolom
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            **Kenapa model ini dipilih?**  
            ✅ **Hasil lebih akurat**: Model ini menggabungkan banyak keputusan sehingga lebih stabil.  
            ✅ **Tidak mudah salah menebak**: Dibandingkan metode lain, model ini lebih bisa mengenali pola anak yang berisiko stunting.  
            ✅ **Sudah diuji dan memberikan hasil terbaik** pada data balita yang digunakan.  
            """
        )

    with col2:
        st.image("random_forest.jpeg", width=300)  # Diagram Random Forest

    # Confusion Matrix
    st.subheader("📊 **Confusion Matrix**")
    cm = confusion_matrix(y_test, rf_model.predict(X_test))
    fig4, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=["Tidak Stunting", "Stunting"], 
                yticklabels=["Tidak Stunting", "Stunting"], ax=ax, linewidths=1, linecolor="black")
    st.pyplot(fig4)

    # Evaluasi Model
    st.subheader("📈 **Evaluasi Model**")
    y_pred_rf = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf)
    recall = recall_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf)

    # Layout hasil evaluasi dengan 4 kolom
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="🎯 **Accuracy**", value=f"{accuracy * 100:.2f} %")
    with col2:
        st.metric(label="🎯 **Precision**", value=f"{precision * 100:.2f} %")
    with col3:
        st.metric(label="🎯 **Recall**", value=f"{recall * 100:.2f} %")
    with col4:
        st.metric(label="🎯 **F1-Score**", value=f"{f1 * 100:.2f} %")

    # Kesimpulan
    st.markdown("### 🔎 Kesimpulan")
    st.info(
        """
        ✅ **Model ini sangat berguna untuk tenaga kesehatan dan orang tua** dalam mengenali potensi stunting pada balita.  
        ✅ **Prediksi yang lebih akurat membantu pengambilan keputusan lebih cepat** dalam pemantauan tumbuh kembang anak.  
        ✅ **Model ini sudah diuji dan dipilih sebagai metode terbaik dalam sistem prediksi ini.**
        """
    )