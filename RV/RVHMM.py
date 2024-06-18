from preprocessing.PreProcesamiento import grabar_audio, preprocesar_audio, extraer_mfcc
from reconocimiento.HMM.Hmm import HMMdict
import pandas as pd

def __entrenarModelo_dict(csv):
    df = pd.read_csv(csv)
    hmmc = HMMdict()
    print("entrenando...")
    for etiqueta in df['etiqueta'].unique():
        caracteristicas = df[df['etiqueta'] == etiqueta]
        carac_mfccs = caracteristicas.drop(columns=['etiqueta']).values
        hmmc.entrenar_hmm(carac_mfccs, etiqueta)
    print("entrenamiento finalizado.")
    return hmmc

def reconocer_voz(duracion=5):
    ruta_grabacion = "grabado/Record1.wav"
    csv = 'caracteristicas_audios.csv'
    modelo = __entrenarModelo_dict(csv)

    #  Pre Procesamiento
    grabar_audio(duracion, nombre=ruta_grabacion)
    mfccs_rec = preprocesar_audio(ruta_grabacion)
    print("Reconociendo...")
    palabras = [ modelo.reconocer_palabra(mfccs) for mfccs in mfccs_rec]
    return palabras
