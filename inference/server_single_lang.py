"""
Single-language TTS Server

This server is designed for users who have downloaded only one or a few language packs.
It loads only the specified language(s) instead of trying to load all languages.

Usage:
    # For Hindi only (default)
    python server_single_lang.py

    # For specific language(s)
    python server_single_lang.py --langs hi
    python server_single_lang.py --langs hi ta  # Multiple languages
    
    # CPU mode (no GPU required)
    python server_single_lang.py --langs hi --cpu
"""

import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.utils.synthesizer import Synthesizer

from src.inference import TextToSpeechEngine
from src.models.request import TTSRequest

LANGUAGE_NAMES = {
    'as' : "Assamese - অসমীয়া",
    'bn' : "Bangla - বাংলা",
    'brx': "Boro - बड़ो",
    'en' : "English (Indian accent)",
    'en+hi': "English+Hindi (Hinglish code-mixed)",
    'gu' : "Gujarati - ગુજરાતી",
    'hi' : "Hindi - हिंदी",
    'kn' : "Kannada - ಕನ್ನಡ",
    'ml' : "Malayalam - മലയാളം",
    'mni': "Manipuri - মিতৈলোন",
    'mr' : "Marathi - मराठी",
    'or' : "Oriya - ଓଡ଼ିଆ",
    'pa' : "Panjabi - ਪੰਜਾਬੀ",
    'raj': "Rajasthani - राजस्थानी",
    'ta' : "Tamil - தமிழ்",
    'te' : "Telugu - తెలుగు",
}


def create_app(languages: list, use_cuda: bool = True):
    """Create FastAPI app with specified languages loaded."""
    
    models = {}
    supported_languages = {}
    
    for lang in languages:
        if lang not in LANGUAGE_NAMES:
            print(f"Warning: Unknown language code '{lang}', skipping...")
            continue
            
        try:
            print(f"Loading model for {lang} ({LANGUAGE_NAMES[lang]})...")
            models[lang] = Synthesizer(
                tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
                tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
                tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
                tts_languages_file=None,
                vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
                vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
                encoder_checkpoint="",
                encoder_config="",
                use_cuda=use_cuda,
            )
            supported_languages[lang] = LANGUAGE_NAMES[lang]
            print(f"✓ Loaded model for {lang}")
            print("*" * 50)
        except Exception as e:
            print(f"✗ Failed to load model for {lang}: {e}")
            print(f"  Make sure you have downloaded the {lang} language pack to checkpoints/{lang}/")
            print("*" * 50)
    
    if not models:
        raise RuntimeError("No models were loaded! Please check your checkpoints directory.")
    
    engine = TextToSpeechEngine(models, enable_denoiser=False)
    
    app = FastAPI(title="AI4Bharat Indic-TTS", description="Text-to-Speech API for Indian Languages")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_class=HTMLResponse)
    def homepage():
        lang_list = "".join([f"<li><code>{code}</code> - {name}</li>" for code, name in supported_languages.items()])
        return f"""
        <html>
        <head>
            <title>AI4Bharat Indic-TTS</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
                code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
                pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                h1 {{ color: #333; }}
                .endpoint {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🗣️ AI4Bharat Indic-TTS API</h1>
            <p>Text-to-Speech API for Indian Languages</p>
            
            <h2>Loaded Languages:</h2>
            <ul>{lang_list}</ul>
            
            <div class="endpoint">
                <h3>POST /</h3>
                <p>Generate speech from text</p>
                <p><strong>Example request:</strong></p>
                <pre>curl -X POST "http://localhost:5050/" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "input": [{{"source": "नमस्ते दुनिया"}}],
    "config": {{
      "gender": "female",
      "language": {{"sourceLanguage": "hi"}}
    }}
  }}'</pre>
            </div>
            
            <div class="endpoint">
                <h3>GET /supported_languages</h3>
                <p>Get list of supported languages</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Health check endpoint</p>
            </div>
        </body>
        </html>
        """
    
    @app.get("/supported_languages")
    def get_supported_languages():
        return supported_languages
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "loaded_languages": list(supported_languages.keys())}
    
    @app.post("/")
    async def batch_tts(request: TTSRequest):
        # Validate that the requested language is loaded
        requested_lang = request.config.language.sourceLanguage
        if requested_lang not in supported_languages:
            return {
                "error": f"Language '{requested_lang}' is not loaded.",
                "available_languages": list(supported_languages.keys()),
                "hint": f"Restart the server with --langs {requested_lang} to load this language."
            }
        return engine.infer_from_request(request)
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Run Indic-TTS server with specific language(s)")
    parser.add_argument(
        "--langs", 
        nargs="+", 
        default=["hi"],
        help="Language code(s) to load. Default: hi (Hindi). Example: --langs hi ta"
    )
    parser.add_argument(
        "--cpu", 
        action="store_true",
        help="Run on CPU instead of GPU"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5050,
        help="Port to run the server on. Default: 5050"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind the server to. Default: 0.0.0.0"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI4Bharat Indic-TTS Server")
    print("=" * 60)
    print(f"Languages to load: {args.langs}")
    print(f"Using GPU: {not args.cpu}")
    print("=" * 60)
    
    app = create_app(languages=args.langs, use_cuda=not args.cpu)
    
    print(f"\n🚀 Starting server at http://{args.host}:{args.port}")
    print(f"   Open http://localhost:{args.port} in your browser for API docs\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
