from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_config import LAB3_RESULTS_PATH, SEED, SENTIMENT_METHODS
from sentiment_datasets import load_sentiment_dataset
from sentiment_methods import build_classical_pipeline, predict_rule, predict_stanza, predict_textblob, predict_transformer
from sentiment_plots import generate_sentiment_wordclouds, plot_class_distribution, plot_compare_results, plot_confusion
from sentiment_training import metric_row, predict_sequence_labels


def compare_sentiment_methods(dataset: str, methods: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    unknown = [method for method in methods if method not in SENTIMENT_METHODS]
    if unknown:
        raise ValueError(f"Nieznane metody: {', '.join(unknown)}")

    df = load_sentiment_dataset(dataset)
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=SEED, stratify=stratify)

    rows: list[dict] = []
    notes: list[str] = []
    files: list[str] = []
    files.extend(generate_sentiment_wordclouds(df, dataset))
    files.append(plot_class_distribution(df, dataset))

    for method in methods:
        try:
            if method in {"nb", "rf"}:
                model = build_classical_pipeline(method)
                model.fit(train_df["text"], train_df["label"])
                predictions = list(model.predict(test_df["text"]))
                model_path = ""
            elif method == "rule":
                predictions = [predict_rule(text).label for text in test_df["text"]]
                model_path = ""
            elif method == "textblob":
                predictions = [predict_textblob(text).label for text in test_df["text"]]
                model_path = ""
            elif method == "stanza":
                predictions = [predict_stanza(text).label for text in test_df["text"]]
                model_path = ""
            elif method == "transformer":
                predictions = [predict_transformer(text).label for text in test_df["text"]]
                model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            else:
                predictions, model_path = predict_sequence_labels(
                    method,
                    dataset,
                    list(test_df["text"]),
                )

            y_true = list(test_df["label"])
            rows.append(metric_row(dataset, method, y_true, predictions, model_path))
            files.append(plot_confusion(y_true, predictions, method, dataset))
        except Exception as exc:
            notes.append(f"{method}: {exc}")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df.to_csv(LAB3_RESULTS_PATH, index=False, encoding="utf-8")
        files.append(str(LAB3_RESULTS_PATH))
        files.append(plot_compare_results(result_df, dataset))
    return result_df, notes, files
