import io

from scipy.io.wavfile import write as scipy_wav_write
from TTS.utils.synthesizer import Synthesizer
from src.inference import TextToSpeechEngine

# Default sampling rate for TTS output (matches model output sample rate from src/inference.py)
DEFAULT_SAMPLING_RATE = 22050

# Initialize Hindi model

lang = "hi"
hi_model  = Synthesizer(
    tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
    tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
    tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
    # tts_speakers_file=None,
    tts_languages_file=None,
    vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
    vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=True,
)

# Initialize Tamil model

lang = "ta"
ta_model  = Synthesizer(
    tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
    tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
    tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
    # tts_speakers_file=None,
    tts_languages_file=None,
    vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
    vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=True,
)

# Setup TTS Engine

models = {
    "hi": hi_model,
    "ta": ta_model,
}
engine = TextToSpeechEngine(models)

# Hindi TTS inference

hindi_raw_audio = engine.infer_from_text(
    input_text="सलाम दुनिया",
    lang="hi",
    speaker_name="male"
)
byte_io = io.BytesIO()
scipy_wav_write(byte_io, DEFAULT_SAMPLING_RATE, hindi_raw_audio)

with open("hindi_audio.wav", "wb") as f:
    f.write(byte_io.getvalue())

# Tamil TTS inference

tamil_raw_audio = engine.infer_from_text(
    input_text="வணக்கம்‌ உலகம்‌",
    lang="ta",
    speaker_name="female"
)
byte_io = io.BytesIO()
scipy_wav_write(byte_io, DEFAULT_SAMPLING_RATE, tamil_raw_audio)

with open("tamil_audio.wav", "wb") as f:
    f.write(byte_io.getvalue())
