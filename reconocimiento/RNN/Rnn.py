import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define una clase para cargar tus datos de audio
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sample_rate = librosa.load(audio_path)
        # Extraer los MFCC del audio y ajustar la longitud a una longitud fija
        mfcc = librosa.feature.mfcc(waveform, sr=sample_rate, n_mfcc=13)
        mfcc = librosa.util.fix_length(mfcc, 500, axis=1)  # Ajustar la longitud a 500
        return mfcc, self.labels[idx]

# Definir el modelo RNN
class VoiceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(VoiceRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Cargar el archivo CSV
csv_file = "../../etiquetas.csv"
df = pd.read_csv(csv_file)

# Crear instancias del conjunto de datos y del cargador de datos
dataset = VoiceDataset(df['ruta_audio'].tolist(), df['etiqueta'].tolist())
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Parámetros del modelo
input_size = 13  # Tamaño de entrada (número de coeficientes MFCC)
hidden_size = 64  # Tamaño del estado oculto de la RNN
num_classes = len(set(df['etiqueta']))  # Número de clases de salida (etiquetas)
num_layers = 2  # Número de capas de la RNN

# Crear modelo
model = VoiceRNN(input_size, hidden_size, num_classes, num_layers)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Función para predecir la palabra a partir de un archivo de audio
def predict_word(audio_file):
    waveform, sample_rate = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(waveform, sr=sample_rate, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, 500, axis=1)  # Ajustar la longitud a 500
    mfcc_tensor = torch.tensor(mfcc.T).unsqueeze(0)  # Convertir a tensor y agregar dimensión de lote
    outputs = model(mfcc_tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Ejemplo de uso
audio_file = "../../samples/uno/uno1.wav"
predicted_word = predict_word(audio_file)
print("Predicted word:", predicted_word)
