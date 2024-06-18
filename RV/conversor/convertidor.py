import os
import librosa
import numpy as np
import pandas as pd
from RV.Dir import SAMPLES, CSV

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error al procesar {file_name}: {e}")
        return None


features_list = []
labels_list = []

for folder_name in os.listdir(SAMPLES):
    folder_path = os.path.join(SAMPLES, folder_name)

    if os.path.isdir(folder_path):  # Verificar si es una carpeta
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                features = extract_features(file_path)

                if features is not None:
                    features_list.append(features)
                    labels_list.append(folder_name)  # La etiqueta es el nombre de la carpeta

# Convertir las listas de características y etiquetas a un DataFrame de pandas
print("creado dataset...")
df = pd.DataFrame(features_list)
df['etiqueta'] = labels_list

df.to_csv(CSV, index=False)
print("dataset creado")
