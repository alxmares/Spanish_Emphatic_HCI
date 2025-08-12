import os
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

from pathlib import Path
import pandas as pd
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
import torch

class EmotionalTTS:
    def __init__(self, model_version="elra-f", model_base_path=None, emotion_csv=None, emotions=None, kinds=None, device="cuda", use_deepspeed=False):
        try:
            self.model_version = model_version
            self.device = device
            self.use_deepspeed = use_deepspeed
            self.latents_cache = {}
            self.model_loaded = False

            # Emociones y tipos según el fine-tuning
            self.emotions = emotions or ["NEUTRAL_NORMAL", "JOY"]
            self.kinds = kinds or ["AFFIRMATIVE", "EXCLAMATORY", "PARAGRAPH", "INTERROGATIVE"]

            # Rutas
            self.model_base_path = Path(model_base_path) if model_base_path else Path.cwd()
            self.emotion_csv = self.model_base_path / self.model_version / (emotion_csv or "inter1sp.csv")

            # Cargar CSV
            print(self.emotion_csv)
            self.df = pd.read_csv(self.emotion_csv)

            # Cargar modelo
            self._load_model()

        except Exception as e:
            raise RuntimeError(f"❌ Error al inicializar EmotionalTTS: {e}")

    def _load_model(self):
        config_path = self.model_base_path / self.model_version / "ready" / "config.json"
        checkpoint_dir = self.model_base_path / self.model_version / "ready"
        speaker_file = checkpoint_dir / "speakers_xtts.pth"
        speaker_file = speaker_file if speaker_file.exists() else None

        config = XttsConfig()
        config.load_json(str(config_path))

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            use_deepspeed=self.use_deepspeed,
            checkpoint_dir=str(checkpoint_dir),
            speaker_file_path=speaker_file
        )
        self.model.to(self.device)
        self.model_loaded = True
        print("✅ Modelo XTTS cargado correctamente.")


    def get_audio_path(self, emotion, kind):
        emotion = "JOY"

        samples = self.df[(self.df["emotion"] == emotion) & (self.df["kind"] == "AFFIRMATIVE")]

        if samples.empty:
            print(f"⚠️ No hay muestra para {emotion}-{kind}, usando cualquier muestra con {emotion}.")
            samples = self.df[self.df["emotion"] == emotion]

        if samples.empty:
            print(f"⚠️ No se encontró ninguna muestra con esa emoción. Se usará cualquier muestra.")
            samples = self.df.copy()

        return samples.sample(1)["audio_path"].iloc[0]

    def get_or_create_latents(self, ref_audio_path):
        if ref_audio_path not in self.latents_cache:
            gpt_latent, speaker_embedding = self.model.get_conditioning_latents(ref_audio_path)
            self.latents_cache[ref_audio_path] = (gpt_latent, speaker_embedding)
        return self.latents_cache[ref_audio_path]

    def synthesize(self, text, language="es", output_path="output.wav",
                   emotion="NEUTRAL_NORMAL", kind="AFFIRMATIVE", speed=1.0, temperature=0.3,
                   length_penalty=1.0, repetition_penalty=1.0, top_k=50, top_p=0.8,
                   ref_audio_path=None):
        if not self.model_loaded:
            raise RuntimeError("Modelo no está cargado.")

        if ref_audio_path is None:
            ref_audio_path = self.get_audio_path(emotion, kind)

        gpt_latent, speaker_embedding = self.get_or_create_latents(ref_audio_path)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            wav_out = self.model.inference(
                text,
                language,
                gpt_cond_latent=gpt_latent.to(torch.float16),
                speaker_embedding=speaker_embedding.to(torch.float16),
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                enable_text_splitting=True,
                speed=speed,
            )

        torchaudio.save(
            output_path,
            torch.tensor(wav_out["wav"], dtype=torch.float32).unsqueeze(0).cpu(),
            sample_rate=24000
        )

        return output_path
