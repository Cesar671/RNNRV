from hmmlearn import hmm
import numpy as np

class HMMdict:

    def __init__(self):
        self.__modelos_hmm = {}

    def entrenar_hmm(self, mfccs, etiqueta, n_components=3, n_iter=100):
        modelo_hmm = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=n_iter)
        modelo_hmm.fit(mfccs)
        self.__modelos_hmm[etiqueta] = modelo_hmm

    def reconocer_palabra(self, mfccs):
        puntajes = {}
        for etiqueta, modelo in self.__modelos_hmm.items():
            puntajes[etiqueta] = modelo.score(mfccs)
        return max(puntajes, key=puntajes.get)

