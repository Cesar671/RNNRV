from RV.preprocessing.PreProcesamiento import grabar_audio, preprocesar_audio
from RV.reconocimiento.HMM.Hmm import HMMdict
import pandas as pd
from RV.Dir import CSV, REC_PROC


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
    modelo = __entrenarModelo_dict(CSV)

    #  Pre Procesamiento
    grabar_audio(duracion, nombre=REC_PROC)
    mfccs_rec = preprocesar_audio(REC_PROC)
    print("Reconociendo...")
    palabras = [modelo.reconocer_palabra(mfccs) for mfccs in mfccs_rec]
    return palabras
