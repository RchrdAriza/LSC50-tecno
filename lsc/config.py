"""LSC50 dataset configuration: sign dictionary and class exclusion.

Central source of truth for the 50 sign classes and any classes we choose to
drop. Reading labels from filenames or Timestamps.xlsx directly scattered
through the pipeline invites inconsistency; import this module instead.
"""
from pathlib import Path

# sign id (4-digit zero-padded string) -> Spanish gloss
SIGNS = {
    "0000": "Gracias",
    "0001": "Buenos Días",
    "0002": "Buenas Tardes",
    "0003": "Buenas Noches",
    "0004": "Seña",
    "0005": "Nombre",
    "0006": "Trabajar",
    "0007": "Comer",
    "0008": "Vivir",
    "0009": "Poco",
    "0010": "Familia",
    "0011": "Personas",
    "0012": "Mujer",
    "0013": "Hombre",
    "0014": "Niño",
    "0015": "Niña",
    "0016": "Abuelo",
    "0017": "Tío",
    "0018": "Hermano",
    "0019": "Hambre",
    "0020": "Feliz",
    "0021": "Contento",
    "0022": "Triste",
    "0023": "Aburrido",
    "0024": "Bien",
    "0025": "Mal",
    "0026": "¿Cómo estas?",
    "0027": "Más o menos",
    "0028": "Sentir",
    "0029": "Jucioso",
    "0030": "Hola",
    "0031": "Adiós",
    "0032": "Por favor",
    "0033": "Con Gusto",
    "0034": "Bienvenido",
    "0035": "Perdón",
    "0036": "Permiso",
    "0037": "Nunca",
    "0038": "Yo",
    "0039": "Tú",
    "0040": "Ustedes",
    "0041": "¿Qué?",
    "0042": "¿Cuándo?",
    "0043": "¿Dónde?",
    "0044": "¿Cómo?",
    "0045": "¿Por qué?",
    "0046": "¿Quién?",
    "0047": "Diferente",
    "0048": "Todos",
    "0049": "Mucho",
}

# Signs excluded from training. Kept as a set for symmetric diff/debugging.
EXCLUDED_SIGNS = {"0049"}

# sign ids (sorted) kept in the dataset after exclusion
KEPT_SIGNS = sorted(set(SIGNS) - EXCLUDED_SIGNS)

# class id -> human label for the KEPT signs, in sorted order
CLASS_LABELS = {i: SIGNS[s] for i, s in enumerate(KEPT_SIGNS)}

DATA_DIR = Path(__file__).resolve().parent.parent

LANDMARK_DIRS = {
    "body": DATA_DIR / "LANDMARKS" / "BODY_LANDMARKS",
    "face": DATA_DIR / "LANDMARKS" / "FACE_LANDMARKS",
    "hand_l": DATA_DIR / "LANDMARKS" / "HANDS_LANDMARKS" / "LEFT_HAND_LANDMARKS",
    "hand_r": DATA_DIR / "LANDMARKS" / "HANDS_LANDMARKS" / "RIGHT_HAND_LANDMARKS",
}


def parse_filename(name: str) -> tuple[str, str, str]:
    """Parse a data filename (with or without extension) into (sign, volunteer, repetition)."""
    base = name if "." not in name else name.rsplit(".", 1)[0]
    sign, vol, rep = base.split("_")
    return sign, vol, rep


def is_included(sign: str) -> bool:
    """True if a sign id should be kept in the pipeline."""
    return sign in SIGNS and sign not in EXCLUDED_SIGNS