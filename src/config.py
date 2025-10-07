from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR/"data"/"audio"
OUTPUT_DIR=BASE_DIR/"data"/"output"
ORCHESTRATOR_OUTPUT_DIR = OUTPUT_DIR/"orchestrator"

GEMINI_MODEL_NAME = "gemini-2.5-pro"
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
FORM_TYPE = Literal["mri","blood_test","undefined"]