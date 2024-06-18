import librosa
import librosa.display
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import medfilt


def extraer_mfcc(ruta_audio, n_mfcc=13):
    y, sr = librosa.load(ruta_audio)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transponer para tener frames x n_mfcc


def preprocesar_audio(audio_path, threshold=30, hop_length=512):
    audio, sr = librosa.load(audio_path)

    audio = librosa.to_mono(audio)

    audio_normalizado = librosa.util.normalize(audio)

    audio_resampled = librosa.resample(audio_normalizado, orig_sr=librosa.get_samplerate(audio_path), target_sr=16000)

    audio_filtrado = medfilt(audio_resampled, kernel_size=3)

    audio_filtrado = np.int16(audio_filtrado * 32767)
    tramos_activos = librosa.effects.split(audio_filtrado, top_db=threshold, hop_length=hop_length)

    audio_sin_espacios, _ = librosa.effects.trim(audio_filtrado)

    audio_sin_espacios = []
    for inicio, fin in tramos_activos:
        audio_sin_espacios.extend(audio_filtrado[inicio:fin])

    audio_sin_espacios = np.array(audio_sin_espacios)
    sf.write("grabado/grabado_pre_pro.wav", audio_sin_espacios, sr)


def grabar_audio(duracion, sr=16000, nombre="grabacion_sample.wav"):
    print("Grabando...")
    audio = sd.rec(int(duracion * sr), samplerate=sr, channels=1)
    sd.wait()  # Espera a que la grabación termine
    print("Grabación completada.")
    sf.write(nombre, audio, sr)
    print(f"Archivo guardado como {nombre}")


grabar_audio(5, nombre="grabado/grabado_raw.wav")
print("preprocesando...")
preprocesar_audio("grabado/grabado_raw.wav")
print("preprocesamiento completado.")
print(f"Archivo guardado como grabado/grabado_pre_pro.wav")
