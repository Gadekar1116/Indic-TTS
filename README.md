# AI4Bharat Indic-TTS

## Towards Building Text-To-Speech Systems for the Next Billion Users

> 🎉 Accepted at ICASSP 2023

Deep learning based text-to-speech (TTS) systems have been evolving rapidly with advances in model architectures, training methodologies, and generalization across speakers and languages. However, these advances have not been thoroughly investigated for Indian language speech synthesis. Such investigation is computationally expensive given the number and diversity of Indian languages, relatively lower resource availability, and the diverse set of advances in neural TTS that remain untested.  In this paper, we evaluate the choice of acoustic models, vocoders, supplementary loss functions, training schedules, and speaker and language diversity for Dravidian and Indo-Aryan languages. Based on this, we identify monolingual models with FastPitch and HiFi-GAN V1, trained jointly on male and female speakers to perform the best. With this setup, we train and evaluate TTS models for 13 languages and find our models to significantly improve upon existing models in all languages as measured by mean opinion scores. We open-source all models on the [Bhashini platform](https://bhashini.gov.in/ulca/model/explore-models).

**TL;DR:** We open-source SOTA Text-To-Speech models for 13 Indian languages: *Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Malayalam, Manipuri, Marathi, Odia, Rajasthani, Tamil and Telugu*.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-assamese-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-assamese-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-bengali-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-bengali-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-bodo-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-bodo-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-gujarati-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-gujarati-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-hindi-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-hindi-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-kannada-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-kannada-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-malayalam-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-malayalam-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-manipuri-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-manipuri-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-marathi-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-marathi-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-rajasthani-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-rajasthani-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-tamil-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-tamil-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-telugu-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-telugu-on-indictts?p=towards-building-text-to-speech-systems-for)
	

**Authors:** Gokul Karthik Kumar*, Praveen S V*, Pratyush Kumar, Mitesh M. Khapra, Karthik Nandakumar

**[[ArXiv Preprint](https://arxiv.org/abs/2211.09536)] [[Audio Samples](https://models.ai4bharat.org/#/tts/samples)] [[Try It Live](https://models.ai4bharat.org/#/tts)] [[Video](https://youtu.be/I3eo8IUAP7s)]**

## Unified architecture of our TTS system
<img src='images/architecture.png' width=1024>

## Results
<img src='images/evaluation.png' width=1024>

## 🚀 Quick Start: Run Locally

If you want to quickly run inference on your laptop without training, follow these steps:

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- ~8GB disk space for models
- (Optional) NVIDIA GPU with CUDA for faster inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/AI4Bharat/Indic-TTS.git
cd Indic-TTS
```

### Step 2: Install System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y libsndfile1-dev ffmpeg enchant
```

**macOS:**
```bash
brew install libsndfile ffmpeg enchant
```

**Windows:**
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH
- Download and install [libsndfile](http://www.mega-nerd.com/libsndfile/)

### Step 3: Create Virtual Environment (Recommended)

> **Note:** Make sure you're inside the `Indic-TTS` directory before proceeding!

```bash
# Verify you're in the correct directory (should show Indic-TTS)
pwd

# Using venv (run from inside the Indic-TTS directory)
python -m venv tts-env
source tts-env/bin/activate  # Linux/macOS
# OR
tts-env\Scripts\activate  # Windows

# OR using conda
conda create -n tts-env python=3.9
conda activate tts-env
```

### Step 4: Install Python Dependencies

> **Note:** You should still be in the `Indic-TTS` directory from Step 1.

```bash
# Navigate to the inference folder (from Indic-TTS root)
cd inference
pip install -r requirements-ml.txt -r requirements-utils.txt
```

### Step 5: Download Pre-trained Models
Download the model checkpoints from the [releases page](https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release).

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download and extract models for the language(s) you need
# Example for Hindi:
wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/hi.zip
unzip hi.zip -d checkpoints/
```

**Available languages:** `as` (Assamese), `bn` (Bengali), `brx` (Bodo), `en` (English), `gu` (Gujarati), `hi` (Hindi), `kn` (Kannada), `ml` (Malayalam), `mni` (Manipuri), `mr` (Marathi), `or` (Odia), `pa` (Punjabi), `raj` (Rajasthani), `ta` (Tamil), `te` (Telugu)

### Step 6: Run Inference

**Option A: Using Python Script**

Create a file `test_tts.py`:
```python
import io
from scipy.io.wavfile import write as scipy_wav_write
from TTS.utils.synthesizer import Synthesizer
from src.inference import TextToSpeechEngine

# Initialize model for your language (e.g., Hindi)
lang = "hi"
model = Synthesizer(
    tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
    tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
    tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
    tts_languages_file=None,
    vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
    vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=False,  # Set to True if you have a GPU
)

# Setup TTS Engine
engine = TextToSpeechEngine(
    models={lang: model},
    enable_denoiser=False  # Set to True for better quality (requires more dependencies)
)

# Generate speech
raw_audio = engine.infer_from_text(
    input_text="नमस्ते, मैं एक टेक्स्ट टू स्पीच सिस्टम हूं।",
    lang=lang,
    speaker_name="female"  # or "male"
)

# Save audio file
byte_io = io.BytesIO()
scipy_wav_write(byte_io, 22050, raw_audio)
with open("output.wav", "wb") as f:
    f.write(byte_io.getvalue())

print("Audio saved to output.wav")
```

Run it:
```bash
python test_tts.py
```

**Option B: Using REST API Server**
```bash
pip install -r requirements-server.txt
python server.py
# Server runs at http://localhost:5050
```

Then send POST requests to generate speech:
```bash
curl -X POST "http://localhost:5050/" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"source": "नमस्ते दुनिया"}],
    "config": {
      "gender": "female",
      "language": {"sourceLanguage": "hi"}
    }
  }'
```

**Option C: Using TTS CLI directly**
```bash
python -m TTS.bin.synthesize \
    --text "नमस्ते दुनिया" \
    --model_path checkpoints/hi/fastpitch/best_model.pth \
    --config_path checkpoints/hi/fastpitch/config.json \
    --vocoder_path checkpoints/hi/hifigan/best_model.pth \
    --vocoder_config_path checkpoints/hi/hifigan/config.json \
    --out_path output.wav
```

### Troubleshooting

**Common Issues:**

1. **ModuleNotFoundError: No module named 'TTS'**
   ```bash
   pip install TTS
   ```

2. **CUDA out of memory**
   - Set `use_cuda=False` in the Synthesizer
   - Or reduce batch size if doing batch inference

3. **Error with enchant/aspell**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install enchant aspell-en
   # macOS
   brew install enchant aspell
   ```

4. **FileNotFoundError for checkpoint files**
   - Ensure models are downloaded and extracted to the correct `checkpoints/<lang>/` directory
   - Check that the directory structure is: `checkpoints/<lang>/fastpitch/` and `checkpoints/<lang>/hifigan/`

5. **librosa/soundfile errors**
   ```bash
   pip install librosa soundfile
   ```

---

## 🏋️ Training Your Own Models

### Environment Setup (for Training):
```
# 1. Create environment
sudo apt-get install libsndfile1-dev ffmpeg enchant
conda create -n tts-env python=3.9
conda activate tts-env

# 2. Setup PyTorch
pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Setup Trainer
git clone https://github.com/gokulkarthik/Trainer 

cd Trainer
pip3 install -e .[all]
cd ..
[or]
cp Trainer/trainer/logging/wandb_logger.py to the local Trainer installation # fixed wandb logger
cp Trainer/trainer/trainer.py to the local Trainer installation # fixed model.module.test_log and added code to log epoch 
add `gpus = [str(gpu) for gpu in gpus]` in line 53 of trainer/distribute.py

# 4. Setup TTS
git clone https://github.com/gokulkarthik/TTS 

cd TTS
pip3 install -e .[all]
cd ..
[or]
cp TTS/TTS/bin/synthesize.py to the local TTS installation # added multiple output support for TTS.bin.synthesis

# 5. Install other requirements
> pip3 install -r requirements.txt
```


### Data Setup:
1. Format IndicTTS dataset in LJSpeech format using [preprocessing/FormatDatasets.ipynb](./preprocessing/FormatDatasets.ipynb)
2. Analyze IndicTTS dataset to check TTS suitability using [preprocessing/AnalyzeDataset.ipynb](./preprocessing/AnalyzeDataset.ipynb)

### Training Steps:
1. Set the configuration with [main.py](./main.py), [vocoder.py](./vocoder.py), [configs](./configs) and [run.sh](./run.sh). Make sure to update the CUDA_VISIBLE_DEVICES in all these files.
2. Train and test by executing `sh run.sh`

---
Code Reference: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
