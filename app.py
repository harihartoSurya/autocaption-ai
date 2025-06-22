import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import tempfile
import os

# ======== SETUP STREAMLIT PAGE =========
st.set_page_config(page_title="🎤 AutoCaption AI", layout="centered")
st.markdown("<h1 style='text-align:center;'>🤖 AutoCaption AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload atau rekam suara lalu dapatkan caption otomatis dan analisis nada bicara!</p>", unsafe_allow_html=True)
st.markdown("---")

# ======== FUNGSIONALITAS ==========

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

def record_voice(duration=5, fs=44100):
    st.info(f"🎙️ Merekam selama {duration} detik... Harap bicara sekarang.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio, fs)
    return temp_file.name

# ======== UI INTERAKTIF ==========

col1, col2 = st.columns(2)

with col1:
    upload = st.file_uploader("📂 Upload file audio (.wav)", type=["wav"])

with col2:
    use_mic = st.button("🎙️ Rekam suara langsung (khusus lokal)")

if upload is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(upload.read())
        audio_path = tmp_file.name
    st.audio(upload, format='audio/wav')
elif use_mic:
    audio_path = record_voice()
    st.audio(audio_path)

else:
    st.info("🔼 Silakan upload file audio atau klik tombol rekam suara.")
    st.stop()

# ======== PROSES DAN OUTPUT ==========

caption = transcribe_audio(audio_path)
st.markdown("### 📝 Caption Otomatis:")
st.success(caption)

pitch_values = analyze_pitch(audio_path)
avg_pitch, tone = interpret_tone(pitch_values)

st.markdown("### 🎵 Analisis Nada Bicara:")
st.markdown(f"- Rata-rata pitch: **{avg_pitch:.2f} Hz**")
st.markdown(f"- Interpretasi: **{tone}**")

st.markdown("### 📈 Visualisasi Nada Bicara:")
fig, ax = plt.subplots()
ax.plot(pitch_values, color='green')
ax.set_title("Kontur Pitch")
ax.set_xlabel("Frame")
ax.set_ylabel("Pitch (Hz)")
st.pyplot(fig)

# Hapus file audio temp
os.remove(audio_path)
