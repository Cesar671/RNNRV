from preprocessing.PreProcesamiento import grabar_audio, preprocesar_audio, extraer_mfcc
from reconocimiento.HMM.Hmm import HMMdict
import pandas as pd
import numpy as np


def __entrenarModelo_dict(csv):
    df = pd.read_csv(csv)
    hmmc = HMMdict()
    for etiqueta in df['etiqueta'].unique():
        rutas = df[df['etiqueta'] == etiqueta]['ruta_audio']
        mfccs = np.concatenate([extraer_mfcc(ruta) for ruta in rutas])
        hmmc.entrenar_hmm(mfccs, etiqueta)
    return hmmc


def reconocer_voz(duracion=5):
    ruta_grabacion = "samples/Record1.wav"
    csv = 'etiquetas.csv'

    modelo = __entrenarModelo_dict(csv)

    #  Pre Procesamiento
    grabar_audio(duracion, nombre=ruta_grabacion)
    preprocesar_audio(ruta_grabacion)
    mfccs_rec = extraer_mfcc(ruta_grabacion)
    #  Reconocimiento
    palabra = modelo.reconocer_palabra(mfccs_rec)
    return palabra
