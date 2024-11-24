import streamlit as st
import joblib
import numpy as np

# Memuat semua model dan scaler dari file
models_and_scaler = joblib.load('prediksi_IQ.pkl')

# Mendapatkan model dan scaler dari dictionary
scaler = models_and_scaler['scaler']
model_IQ = models_and_scaler['model_IQ']
model_description = models_and_scaler['model_description']

# Desain aplikasi Streamlit yang minimalis dan elegan
st.set_page_config(page_title="Test IQ", page_icon="🧠", layout="centered")

# Judul Aplikasi dengan gaya elegan
st.markdown("""
    <h1 style="text-align: center; color: #FF6347;">Prediksi IQ</h1>
    <p style="text-align: center; font-size: 20px; color: #333;">Masukkan skor mentah untuk melihat hasil prediksi IQ dan kategorinya.</p>
""", unsafe_allow_html=True)

# Styling input box yang lebih kecil dan kompak
input_style = """
    <style>
    .stNumberInput>div>div>input {
        width: 80px;  /* Lebar input box lebih kecil */
        height: 30px;  /* Tinggi input box lebih kecil */
        font-size: 18px;
        text-align: center;
        border-radius: 5px;
        border: 2px solid #FF6347;
        margin: 0 auto;
    }
    .stNumberInput>div>label {
        font-size: 16px;
        font-weight: bold;
        color: #FF6347;
    }
    </style>
"""
st.markdown(input_style, unsafe_allow_html=True)

# Input skor mentah
raw_score_input = st.number_input("Masukkan Skor Mentah:", min_value=0, max_value=100, step=1, value=50)

# Styling untuk tombol prediksi yang lebih elegan
button_style = """
    <style>
    .stButton>button {
        background-color: #FF6347;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        width: 150px;  /* Lebar tombol lebih kecil */
        margin: 20px auto;
        display: block;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF4500;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Fungsi prediksi
def predict_IQ_and_description(raw_score):
    # Transformasi input skor mentah menggunakan scaler
    scaled_score = scaler.transform([[raw_score]])
    
    # Prediksi nilai IQ dan deskripsi
    predicted_IQ = model_IQ.predict(scaled_score)[0]
    predicted_description = model_description.predict(scaled_score)[0]
    
    return predicted_IQ, predicted_description

# Menambahkan tips per kategori deskripsi
def get_tips(description):
    tips = {
        "Di Bawah Rata-Rata": [
            "🔴 Cobalah lebih banyak latihan soal untuk meningkatkan kemampuan kognitif.",
            "📖 Belajar secara rutin dapat membantu meningkatkan skor Anda.",
            "🧠 Cobalah untuk melakukan latihan otak seperti teka-teki dan puzzle."
        ],
        "Rata-Rata": [
            "📊 Teruskan usaha Anda! Anda berada di jalur yang baik.",
            "💡 Lakukan latihan untuk memperkuat konsep yang sudah dikuasai.",
            "🔍 Tingkatkan fokus dan strategi dalam memecahkan soal."
        ],
        "Di Atas Rata-Rata": [
            "🌟 Sangat baik! Cobalah untuk terus mengasah keterampilan Anda lebih lanjut.",
            "🚀 Tantang diri Anda dengan soal-soal yang lebih sulit.",
            "🧩 Jangan berhenti di sini, kembangkan kemampuan Anda dengan berbagai metode."
        ]
    }
    return tips.get(description, ["💭 Tidak ada tips yang tersedia."])

# Ikon untuk kategori deskripsi IQ
def get_icon(description):
    icons = {
        "Di Bawah Rata-Rata": "🔻",
        "Rata-Rata": "📊",
        "Di Atas Rata-Rata": "🌟",
    }
    return icons.get(description, "⚙️")

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    if raw_score_input:
        predicted_IQ, predicted_description = predict_IQ_and_description(raw_score_input)
        
        # Menampilkan hasil prediksi dengan gaya visual elegan
        icon = get_icon(predicted_description)
        tips = get_tips(predicted_description)
        
        # Hasil Prediksi dengan styling yang bersih dan berwarna
        st.markdown(f"""
            <h2 style="text-align: center; color: #32CD32;">Hasil Prediksi</h2>
            <p style="text-align: center; font-size: 24px; color: #333;">
                <strong>Prediksi Nilai IQ:</strong> <span style="color: #FF6347;">{predicted_IQ:.2f}</span><br>
                <strong>Keterangan:</strong> <span style="color: #FF6347;">{predicted_description}</span> {icon}
            </p>
            <h4 style="text-align: center; color: #FF1493;">Tips untuk Anda:</h4>
            <ul style="text-align: center; font-size: 18px; color: #555;">
                {''.join([f"<li style='color: #FF6347;'>{tip}</li>" for tip in tips])}
            </ul>
        """, unsafe_allow_html=True)
    else:
        st.warning("Silakan masukkan skor mentah terlebih dahulu.")
