import joblib
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np
import os


class SERModel:
    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 wav2vec_model: str = "facebook/wav2vec2-large-xlsr-53-spanish",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Inicializa y carga todos los componentes del pipeline SER.
        """
        self.device = device

        # Cargar clasificadores Y scaler
        self.ser_model = joblib.load(model_path)
        self.ser_scaler = joblib.load(scaler_path)

        # Cargar Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model)
        self.wav2vec_model.eval().to(self.device)

        print("✅ Modelo SER cargado con éxito.")

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extrae embeddings de la capa 6 del modelo Wav2Vec2 y retorna un vector transformado.
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resamplear si no es 16 kHz
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Asegurarse de que es MONO
        waveform = waveform.squeeze(0)  # Mono
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs, output_hidden_states=True)
            features = torch.mean(outputs.hidden_states[6], dim=1).squeeze(0).cpu().numpy()

        return features
    
    def predict(self, audio_path: str) -> dict:
        """
        Procesa un archivo de audio y devuelve las predicciones del modelo SER con salidas binarias por etiqueta.
        """
        # Listas de clases por tarea
        perfiles = ['Child', 'Female', 'Male']
        regiones = ['Spain', 'Mex']
        emociones = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

        features = self.extract_features(audio_path)
        features_scaled = self.ser_scaler.transform([features])
        pred_probs = self.ser_model.predict_proba(features_scaled)
        def get_tarea_probs(indices, etiquetas):
            # Extraer la probabilidad de clase positiva (índice 1) para cada etiqueta binaria
            probs = [float(pred_probs[i][0][1]) for i in indices]
            idx_max = int(np.argmax(probs))
            return etiquetas[idx_max], dict(zip(etiquetas, probs))

        clase_1, probas_1 = get_tarea_probs([7, 8, 9], perfiles)
        clase_2, probas_2 = get_tarea_probs([10, 11], regiones)
        clase_3, probas_3 = get_tarea_probs([0, 1, 2, 3, 4, 5, 6], emociones)

        return {
            "tarea_1": {"clase": clase_1, "probabilidades": probas_1},
            "tarea_2": {"clase": clase_2, "probabilidades": probas_2},
            "tarea_3": {"clase": clase_3, "probabilidades": probas_3}
        }
