"""ASR engine placeholder — model loading to be implemented."""
import logging

logger = logging.getLogger(__name__)

_model = None


def load_model() -> None:
    """Load ASR model. Placeholder — replace with actual model initialization."""
    global _model
    logger.info("ASR model loading placeholder called. No model loaded.")
    # TODO: load actual ASR model here
    _model = None


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw PCM audio bytes to text. Placeholder implementation."""
    # TODO: replace with actual inference
    return ""
