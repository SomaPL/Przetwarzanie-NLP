from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sentiment_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_LEN,
    DEFAULT_VOCAB_SIZE,
    LAB3_PLOTS_DIR,
    MODELS_DIR,
    SEED,
)
from sentiment_datasets import load_sentiment_dataset
from sentiment_methods import SentimentPrediction
from sentiment_plots import plot_confusion, plot_training_history


warnings.filterwarnings("ignore", message=".*HDF5 file.*")
warnings.filterwarnings("ignore", message=".*legacy.*")


@dataclass
class TrainingResult:
    model_name: str
    dataset: str
    accuracy: float
    macro_f1: float
    model_path: str
    tokenizer_path: str
    encoder_path: str
    history_plot: str
    confusion_plot: str
    seconds: float
    epochs: int
    max_len: int

    def as_message(self) -> str:
        return (
            f"Model: {self.model_name.upper()}\n"
            f"Dataset: {self.dataset}\n"
            f"Accuracy: {self.accuracy:.3f}\n"
            f"Macro F1: {self.macro_f1:.3f}\n"
            f"Epoki: {self.epochs}, max_len: {self.max_len}\n"
            f"Czas: {self.seconds:.1f}s\n"
            f"Model: {self.model_path}\n"
            f"Tokenizer: {self.tokenizer_path}\n"
            f"Encoder: {self.encoder_path}\n"
            f"Wykres: {self.history_plot}"
        )


def train_sequence_model(
    model_name: str,
    dataset: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_len: int = DEFAULT_MAX_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> TrainingResult:
    model_name = model_name.lower().strip()
    if model_name not in {"simplernn", "lstm", "gru"}:
        raise ValueError("Model musi mieć wartość: simplernn, lstm albo gru.")

    tf, Tokenizer, pad_sequences, layers, models, callbacks = _load_tensorflow()

    MODELS_DIR.mkdir(exist_ok=True)
    LAB3_PLOTS_DIR.mkdir(exist_ok=True)

    df = load_sentiment_dataset(dataset)
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        df["text"].astype(str),
        df["label"].astype(str),
        test_size=0.25,
        random_state=SEED,
        stratify=stratify,
    )

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(x_train)
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding="post", truncating="post")
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding="post", truncating="post")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    model = build_sequence_network(
        model_name=model_name,
        layers=layers,
        models=models,
        vocab_size=min(vocab_size, len(tokenizer.word_index) + 2),
        embedding_dim=embedding_dim,
        max_len=max_len,
        class_count=len(encoder.classes_),
    )

    patience = max(1, epochs // 10)
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    start = time.perf_counter()
    history = model.fit(
        x_train_pad,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    seconds = time.perf_counter() - start

    probabilities = model.predict(x_test_pad, verbose=0)
    predicted_ids = probabilities.argmax(axis=1)
    predicted_labels = encoder.inverse_transform(predicted_ids)

    accuracy = accuracy_score(y_test_raw, predicted_labels)
    macro_f1 = f1_score(y_test_raw, predicted_labels, average="macro", zero_division=0)

    paths = model_paths(model_name, dataset)
    model.save(paths["model"])
    save_pickle(tokenizer, paths["tokenizer"])
    save_pickle(encoder, paths["encoder"])
    paths["meta"].write_text(
        json.dumps(
            {
                "model": model_name,
                "dataset": dataset,
                "max_len": max_len,
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "classes": encoder.classes_.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    history_plot = plot_training_history(history.history, model_name, dataset)
    confusion_plot = plot_confusion(list(y_test_raw), list(predicted_labels), model_name, dataset)

    return TrainingResult(
        model_name=model_name,
        dataset=dataset,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        model_path=str(paths["model"]),
        tokenizer_path=str(paths["tokenizer"]),
        encoder_path=str(paths["encoder"]),
        history_plot=history_plot,
        confusion_plot=confusion_plot,
        seconds=seconds,
        epochs=epochs,
        max_len=max_len,
    )


def build_sequence_network(
    model_name: str,
    layers,
    models,
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    class_count: int,
):
    recurrent_layer = {
        "simplernn": layers.SimpleRNN(64),
        "lstm": layers.LSTM(64),
        "gru": layers.GRU(64),
    }[model_name]

    model = models.Sequential(
        [
            layers.Input(shape=(max_len,)),
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            recurrent_layer,
            layers.Dense(64, activation="relu"),
            layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_with_sequence_model(model_name: str, dataset: str, text: str) -> SentimentPrediction:
    tf, _Tokenizer, pad_sequences, _layers, models, _callbacks = _load_tensorflow()
    paths = model_paths(model_name, dataset)
    missing = [str(path) for key, path in paths.items() if key != "meta" and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Brakuje zapisanego modelu. Najpierw uruchom np. "
            f"/train model={model_name} dataset={dataset}. Brakujące pliki: {', '.join(missing)}"
        )

    model = models.load_model(paths["model"])
    tokenizer = load_pickle(paths["tokenizer"])
    encoder = load_pickle(paths["encoder"])
    meta = load_meta(paths["meta"])
    max_len = int(meta.get("max_len", DEFAULT_MAX_LEN))

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    best_id = int(np.argmax(probabilities))
    label = str(encoder.inverse_transform([best_id])[0])
    score = float(probabilities[best_id])
    return SentimentPrediction(model_name, label, score, "model sekwencyjny z pliku .h5", str(paths["model"]))


def evaluate_sequence_model(model_name: str, dataset: str, texts: list[str], labels: list[str]) -> dict:
    predicted_labels, model_path = predict_sequence_labels(model_name, dataset, texts)
    return metric_row(dataset, model_name, labels, predicted_labels, model_path)


def predict_sequence_labels(model_name: str, dataset: str, texts: list[str]) -> tuple[list[str], str]:
    tf, _Tokenizer, pad_sequences, _layers, models, _callbacks = _load_tensorflow()
    paths = model_paths(model_name, dataset)
    if not paths["model"].exists():
        raise FileNotFoundError(f"Brak modelu {paths['model']}. Uruchom /train model={model_name} dataset={dataset}.")

    model = models.load_model(paths["model"])
    tokenizer = load_pickle(paths["tokenizer"])
    encoder = load_pickle(paths["encoder"])
    meta = load_meta(paths["meta"])
    max_len = int(meta.get("max_len", DEFAULT_MAX_LEN))

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)
    predicted_labels = encoder.inverse_transform(probabilities.argmax(axis=1))
    return list(predicted_labels), str(paths["model"])


def metric_row(dataset: str, method: str, y_true: list[str], y_pred: list[str], model_path: str = "") -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "model_path": model_path,
    }


def model_paths(model_name: str, dataset: str) -> dict[str, Path]:
    stem = f"{model_name}_{dataset}"
    return {
        "model": MODELS_DIR / f"{stem}.h5",
        "tokenizer": MODELS_DIR / f"{stem}_tokenizer.h5",
        "encoder": MODELS_DIR / f"{stem}_label_encoder.h5",
        "meta": MODELS_DIR / f"{stem}_meta.json",
    }


def list_saved_models() -> list[dict[str, str]]:
    MODELS_DIR.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for model_path in sorted(MODELS_DIR.glob("*.h5")):
        if model_path.stem.endswith("_tokenizer") or model_path.stem.endswith("_label_encoder"):
            continue
        parts = model_path.stem.split("_", 1)
        model_name = parts[0]
        dataset = parts[1] if len(parts) > 1 else "unknown"
        paths = model_paths(model_name, dataset)
        rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                "path": str(model_path),
                "tokenizer": "tak" if paths["tokenizer"].exists() else "nie",
                "encoder": "tak" if paths["encoder"].exists() else "nie",
            }
        )
    return rows


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks, layers, models
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Brakuje TensorFlow/Keras. Uruchom: pip install -r requirements.txt") from exc

    tf.get_logger().setLevel("ERROR")
    return tf, Tokenizer, pad_sequences, layers, models, callbacks
