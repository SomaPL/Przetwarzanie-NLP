from pathlib import Path


SENTIMENT_LABELS = ["negatywny", "neutralny", "pozytywny"]
SEED = 42

CUSTOM_DATASET_PATH = Path("sentiment_dataset.csv")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LAB3_PLOTS_DIR = Path("lab3plots")
LAB3_RESULTS_PATH = Path("lab3results.csv")

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 200
DEFAULT_VOCAB_SIZE = 10000
DEFAULT_EMBEDDING_DIM = 100

SEQUENCE_MODELS = ["simplernn", "lstm", "gru"]
SENTIMENT_METHODS = [
    "rule",
    "nb",
    "rf",
    "transformer",
    "textblob",
    "stanza",
    "simplernn",
    "lstm",
    "gru",
]

