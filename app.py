import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import tempfile
import os

st.set_page_config(page_title="AutoCaption AI", layout="centered")

st.title("🎤 AutoCaption AI (Web Version)")
st.markdown("Upload file audio `.wav`, dapatkan **caption otomatis** dan **analisis nada bicara**.")

# Upload audio
uploaded_file = st.file_uploader("📂 Upload file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    # Fungsi transkripsi
    def transcribe_audio(audio_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language='id-ID')
        except sr.UnknownValueError:
            return "❌ Tidak bisa mengenali ucapan."
        except sr.RequestError:
            return "❌ Gagal terhubung ke layanan transkripsi."

    # Fungsi pitch
    def analyze_pitch(audio_path):
        y, sr_ = librosa.load(audio_path)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_)
        pitch_values = [
            pitches[magnitudes[:, i].argmax(), i]
            for i in range(pitches.shape[1])
            if pitches[magnitudes[:, i].argmax(), i] > 0
        ]
        return pitch_values

    def interpret_tone(pitch_values):
        avg_pitch = np.mean(pitch_values)
        if avg_pitch > 200:
            tone = "Nada tinggi (semangat / bertanya)"
        elif avg_pitch > 100:
            tone = "Nada normal (pernyataan)"
        else:
            tone = "Nada rendah (serius / sedih)"
        return avg_pitch, tone

    # Caption
    caption = transcribe_audio(audio_path)
    st.subheader("📝 Caption Otomatis")
    st.write(caption)

    # Pitch
    pitch_values = analyze_pitch(audio_path)
    avg_pitch, tone = interpret_tone(pitch_values)

    st.subheader("🎵 Analisis Nada Bicara")
    st.markdown(f"- Rata-rata pitch: `{avg_pitch:.2f} Hz`")
    st.markdown(f"- Interpretasi: **{tone}**")

    # Grafik
    st.subheader("📈 Visualisasi Pitch")
    fig, ax = plt.subplots()
    ax.plot(pitch_values, color='green')
    ax.set_title("Kontur Nada Bicara")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pitch (Hz)")
    ax.grid(True)
    st.pyplot(fig)

    # Hapus file temp
    os.remove(audio_path)
