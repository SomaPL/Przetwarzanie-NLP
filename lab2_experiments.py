from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from wordcloud import WordCloud


warnings.filterwarnings("ignore", category=ConvergenceWarning)

PLOTS_DIR = Path("lab2plots")
RESULTS_PATH = Path("lab2results.csv")
SUMMARY_PATH = Path("lab2results_summary.csv")
FEATURE_IMPORTANCE_PATH = Path("lab2_feature_importance.txt")
SIMILAR_WORDS_PATH = Path("lab2_similar_words.txt")
GLOVE_PATH = Path("data/glove.6B.50d.txt")

SEEDS = [42, 1337, 2024]
ALL_MODELS = ["nb", "rf", "mlp", "logreg"]
ALL_EMBEDDINGS = ["bow", "tfidf", "word2vec", "glove"]
QUERY_WORDS = ["space", "computer", "science", "music", "car"]


@dataclass(frozen=True)
class TextDataset:
    name: str
    texts: list[str]
    labels: list[str]


@dataclass(frozen=True)
class ExperimentSettings:
    dataset: str
    methods: list[str]
    gridsearch: bool
    runs: int
    embeddings: list[str]
    max_features: int = 5000
    max_plot_points: int = 800


@dataclass
class PreparedVectors:
    x_train: Any
    x_test: Any
    x_all: Any
    feature_names: list[str]
    word_vectors: dict[str, np.ndarray]
    notes: list[str]


@dataclass
class ExperimentReport:
    dataset: str
    rows: list[dict[str, Any]]
    notes: list[str]
    files: list[str]

    def as_message(self) -> str:
        if not self.rows:
            return "Eksperyment nie zwrócił wyników."

        df = pd.DataFrame(self.rows)
        summary = (
            df.groupby(["embedding", "model"], as_index=False)
            .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"))
            .sort_values(["macro_f1", "accuracy"], ascending=False)
        )

        lines = [
            f"Gotowe: dataset={self.dataset}",
            "",
            "Najlepsze średnie wyniki:",
        ]
        for row in summary.head(8).itertuples(index=False):
            lines.append(
                f"- {row.embedding}/{row.model}: "
                f"accuracy={row.accuracy:.3f}, macro_f1={row.macro_f1:.3f}"
            )

        lines.extend(
            [
                "",
                "Pliki:",
                f"- {RESULTS_PATH}",
                f"- {SUMMARY_PATH}",
                f"- {FEATURE_IMPORTANCE_PATH}",
                f"- {SIMILAR_WORDS_PATH}",
                f"- {PLOTS_DIR}/",
            ]
        )

        if self.notes:
            lines.append("")
            lines.append("Uwagi:")
            lines.extend(f"- {note}" for note in dict.fromkeys(self.notes))

        return "\n".join(lines)[:3900]


def parse_classify_command(text: str) -> ExperimentSettings:
    tokens = text.split()
    params: dict[str, str] = {}

    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        params[key.strip().lower()] = value.strip()

    dataset = params.get("dataset")
    if not dataset:
        raise ValueError(
            "Brakuje parametru dataset. Przykład: "
            "/classify dataset=20news_group method=logreg gridsearch=false run=1"
        )

    method_value = params.get("method", "all").lower()
    if method_value == "all":
        methods = ALL_MODELS.copy()
    else:
        methods = [m.strip() for m in method_value.split(",") if m.strip()]

    unknown_methods = [m for m in methods if m not in ALL_MODELS]
    if unknown_methods:
        raise ValueError(f"Nieznany model: {', '.join(unknown_methods)}")

    gridsearch = params.get("gridsearch", "false").lower() == "true"

    try:
        runs = int(params.get("run", "1"))
    except ValueError as exc:
        raise ValueError("Parametr run musi być liczbą: 1, 2 albo 3") from exc

    if runs < 1 or runs > 3:
        raise ValueError("Parametr run musi mieć wartość 1, 2 albo 3")

    embedding_value = params.get("embedding", "all").lower()
    if embedding_value == "all":
        embeddings = ALL_EMBEDDINGS.copy()
    else:
        embeddings = [e.strip() for e in embedding_value.split(",") if e.strip()]

    unknown_embeddings = [e for e in embeddings if e not in ALL_EMBEDDINGS]
    if unknown_embeddings:
        raise ValueError(f"Nieznany embedding: {', '.join(unknown_embeddings)}")

    return ExperimentSettings(
        dataset=dataset,
        methods=methods,
        gridsearch=gridsearch,
        runs=runs,
        embeddings=embeddings,
    )


def run_dataset_experiment(config: ExperimentSettings) -> ExperimentReport:
    PLOTS_DIR.mkdir(exist_ok=True)

    dataset = load_dataset(config.dataset)
    notes: list[str] = []
    files: list[str] = []

    wordcloud_paths = generate_wordclouds(dataset)
    files.extend(wordcloud_paths)

    rows: list[dict[str, Any]] = []
    feature_blocks: list[str] = []
    confusion_values: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: {"true": [], "pred": []}
    )
    visualized_docs: set[tuple[str, str]] = set()
    visualized_words: set[str] = set()
    similar_blocks: list[str] = []

    labels_order = sorted(set(dataset.labels))

    for seed in SEEDS[: config.runs]:
        stratify = _stratify_labels(dataset.labels)
        x_train_texts, x_test_texts, y_train, y_test = train_test_split(
            dataset.texts,
            dataset.labels,
            test_size=_test_size(dataset.labels),
            random_state=seed,
            stratify=stratify,
        )

        for embedding in config.embeddings:
            representation = build_representation(
                embedding=embedding,
                train_texts=x_train_texts,
                test_texts=x_test_texts,
                all_texts=dataset.texts,
                seed=seed,
                max_features=config.max_features,
            )
            notes.extend(representation.notes)

            if embedding in {"word2vec", "glove"} and embedding not in visualized_words:
                similar_blocks.append(
                    format_similar_words(embedding, representation.word_vectors)
                )
                files.extend(plot_word_embeddings(embedding, representation.word_vectors, seed))
                visualized_words.add(embedding)

            for model_name in config.methods:
                estimator, param_grid = make_estimator(model_name, embedding, seed)

                if config.gridsearch:
                    cv = _grid_cv(y_train, seed)
                    if cv is None:
                        notes.append(
                            "GridSearchCV pominięty dla części modeli, bo klasy mają za mało próbek."
                        )
                    else:
                        estimator = GridSearchCV(
                            estimator,
                            param_grid=param_grid,
                            scoring="f1_macro",
                            cv=cv,
                            n_jobs=1,
                        )

                estimator.fit(representation.x_train, y_train)
                predictions = estimator.predict(representation.x_test)

                accuracy = accuracy_score(y_test, predictions)
                macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
                best_params = getattr(estimator, "best_params_", None)

                rows.append(
                    {
                        "dataset": dataset.name,
                        "embedding": embedding,
                        "model": model_name,
                        "accuracy": round(float(accuracy), 6),
                        "macro_f1": round(float(macro_f1), 6),
                        "seed": seed,
                        "gridsearch": config.gridsearch,
                        "best_params": best_params or "",
                    }
                )

                key = (embedding, model_name)
                confusion_values[key]["true"].extend(y_test)
                confusion_values[key]["pred"].extend(predictions)

                if key not in visualized_docs:
                    files.extend(
                        plot_document_embeddings(
                            x=representation.x_all,
                            labels=dataset.labels,
                            dataset_name=dataset.name,
                            model_name=model_name,
                            embedding=embedding,
                            seed=seed,
                            max_points=config.max_plot_points,
                        )
                    )
                    visualized_docs.add(key)

                feature_blocks.extend(
                    feature_importance_blocks(
                        estimator=estimator,
                        model_name=model_name,
                        embedding=embedding,
                        dataset_name=dataset.name,
                        feature_names=representation.feature_names,
                    )
                )

    files.extend(plot_confusions(confusion_values, labels_order))
    write_results(rows)
    write_feature_importance(feature_blocks)
    write_similar_words(similar_blocks)

    return ExperimentReport(dataset=dataset.name, rows=rows, notes=notes, files=files)


def load_dataset(dataset_name: str) -> TextDataset:
    raw_name = dataset_name.strip()
    normalized = raw_name.lower().strip()

    if normalized in {"demo", "sample", "test"}:
        return demo_dataset()

    if "20news" in normalized or "20newsgroup" in normalized:
        data = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
        )
        labels = [data.target_names[target] for target in data.target]
        return TextDataset(name="20news_group", texts=list(data.data), labels=labels)

    if "imdb" in normalized:
        return load_local_dataset(
            name="imdb",
            paths=[
                Path("data/imdb.csv"),
                Path("data/IMDB Dataset.csv"),
                Path("data/imdb-dataset.csv"),
            ],
        )

    if "amazon" in normalized:
        return load_local_dataset(
            name="amazon",
            paths=[
                Path("data/amazon.csv"),
                Path("data/amazon_reviews.csv"),
                Path("data/Reviews.csv"),
            ],
        )

    if "ag_news" in normalized or "ag-news" in normalized or normalized == "agnews":
        return load_local_dataset(
            name="ag_news",
            paths=[
                Path("data/ag_news.csv"),
                Path("data/ag-news.csv"),
                Path("data/agnews.csv"),
            ],
        )

    path = Path(raw_name)
    if path.exists():
        return load_local_dataset(name=path.stem, paths=[path])

    raise ValueError(
        "Nie znam datasetu. Dostępne: 20news_group, demo albo lokalne pliki CSV "
        "dla imdb/amazon/ag_news w katalogu data/."
    )


def load_local_dataset(name: str, paths: list[Path]) -> TextDataset:
    existing = next((path for path in paths if path.exists()), None)
    if existing is None:
        expected = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(
            f"Brak lokalnego pliku dla dataset={name}. Umieść jeden z plików: {expected}"
        )

    df = _read_csv(existing)
    if df.empty:
        raise ValueError(f"Plik {existing} jest pusty.")

    text_col = _find_column(
        df,
        ["review", "text", "content", "sentence", "comment", "description", "title"],
    )
    label_col = _find_column(
        df,
        ["sentiment", "label", "class", "target", "category", "overall", "score", "rating"],
    )

    if text_col is None or label_col is None:
        raise ValueError(
            f"Nie mogę znaleźć kolumn text/label w {existing}. "
            "Nazwij kolumny np. text,label albo review,sentiment."
        )

    clean_df = df[[text_col, label_col]].dropna()
    texts = clean_df[text_col].astype(str).tolist()
    labels = [_normalize_label(value, label_col) for value in clean_df[label_col].tolist()]

    pairs = [(text.strip(), label) for text, label in zip(texts, labels) if text.strip()]
    if len(pairs) < 4:
        raise ValueError("Dataset ma za mało poprawnych rekordów.")

    texts, labels = zip(*pairs)
    return TextDataset(name=name, texts=list(texts), labels=list(labels))


def demo_dataset() -> TextDataset:
    texts = [
        "Space missions use rockets and satellites for science.",
        "The telescope found a planet near a distant star.",
        "Astronauts train for orbital flights and space walks.",
        "Computer graphics require fast processors and memory.",
        "The software update fixed a network security bug.",
        "A new database server improved application performance.",
        "The team won the match after a strong second half.",
        "Fans watched the football game at the stadium.",
        "The coach changed tactics before the final tournament.",
        "Electric cars need batteries and charging stations.",
        "The engine repair made the car much quieter.",
        "Drivers compare fuel economy and road safety features.",
    ]
    labels = [
        "space",
        "space",
        "space",
        "computer",
        "computer",
        "computer",
        "sport",
        "sport",
        "sport",
        "car",
        "car",
        "car",
    ]
    return TextDataset(name="demo", texts=texts, labels=labels)


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1250", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {str(col).lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for col in df.columns:
        col_lower = str(col).lower()
        if any(candidate in col_lower for candidate in candidates):
            return col

    return None


def _normalize_label(value: Any, column_name: str) -> str:
    if pd.isna(value):
        return ""

    rating_columns = {"overall", "score", "rating", "stars"}
    if str(column_name).lower().strip() in rating_columns:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).strip().lower()
        if number <= 2:
            return "negative"
        if number >= 4:
            return "positive"
        return "neutral"

    return str(value).strip().lower()


def build_representation(
    embedding: str,
    train_texts: list[str],
    test_texts: list[str],
    all_texts: list[str],
    seed: int,
    max_features: int,
) -> PreparedVectors:
    if embedding == "bow":
        vectorizer = CountVectorizer(max_features=max_features)
        x_train = vectorizer.fit_transform(train_texts)
        return PreparedVectors(
            x_train=x_train,
            x_test=vectorizer.transform(test_texts),
            x_all=vectorizer.transform(all_texts),
            feature_names=vectorizer.get_feature_names_out().tolist(),
            word_vectors={},
            notes=[],
        )

    if embedding == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train = vectorizer.fit_transform(train_texts)
        return PreparedVectors(
            x_train=x_train,
            x_test=vectorizer.transform(test_texts),
            x_all=vectorizer.transform(all_texts),
            feature_names=vectorizer.get_feature_names_out().tolist(),
            word_vectors={},
            notes=[],
        )

    word_vectors, notes = fit_word_vectors(
        texts=train_texts,
        embedding=embedding,
        seed=seed,
        max_features=max_features,
    )
    dim = len(next(iter(word_vectors.values()))) if word_vectors else 50
    feature_names = [f"dim_{i}" for i in range(dim)]
    return PreparedVectors(
        x_train=texts_to_average_vectors(train_texts, word_vectors, dim),
        x_test=texts_to_average_vectors(test_texts, word_vectors, dim),
        x_all=texts_to_average_vectors(all_texts, word_vectors, dim),
        feature_names=feature_names,
        word_vectors=word_vectors,
        notes=notes,
    )


def fit_word_vectors(
    texts: list[str],
    embedding: str,
    seed: int,
    max_features: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    notes: list[str] = []

    if embedding == "glove" and GLOVE_PATH.exists():
        vocabulary = _top_vocabulary(texts, max_features)
        vectors = load_glove_vectors(GLOVE_PATH, vocabulary)
        if vectors:
            return vectors, notes
        notes.append("Plik GloVe istnieje, ale nie znaleziono slow z korpusu.")

    vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w\w+\b")
    term_counts = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    if embedding == "glove":
        notes.append(
            "Nie znaleziono data/glove.6B.50d.txt, więc użyto lokalnej reprezentacji na log-zliczeniach i SVD."
        )
        matrix = term_counts.astype(float).tocsr()
        matrix.data = np.log1p(matrix.data)
        matrix = matrix.T
    else:
        matrix = term_counts.T.astype(float)

    max_components = min(matrix.shape) - 1
    n_components = min(50, max(2, max_components))
    if max_components < 2:
        rng = np.random.default_rng(seed)
        return {
            word: rng.normal(0, 0.01, size=2).astype(float)
            for word in feature_names
        }, notes

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    word_matrix = svd.fit_transform(matrix)
    return {
        word: word_matrix[index].astype(float)
        for index, word in enumerate(feature_names)
    }, notes


def load_glove_vectors(path: Path, vocabulary: set[str]) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            word = parts[0]
            if word not in vocabulary:
                continue
            try:
                vectors[word] = np.array([float(x) for x in parts[1:]], dtype=float)
            except ValueError:
                continue
    return vectors


def texts_to_average_vectors(
    texts: list[str],
    word_vectors: dict[str, np.ndarray],
    dim: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    zero = np.zeros(dim, dtype=float)

    for text in texts:
        vectors = [word_vectors[token] for token in tokenize_text(text) if token in word_vectors]
        if vectors:
            rows.append(np.mean(vectors, axis=0))
        else:
            rows.append(zero.copy())

    return np.vstack(rows)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


def _top_vocabulary(texts: list[str], max_features: int) -> set[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_text(text))
    return {word for word, _ in counter.most_common(max_features)}


def make_estimator(model_name: str, embedding: str, seed: int) -> tuple[Any, dict[str, list[Any]]]:
    dense_embedding = embedding in {"word2vec", "glove"}

    if model_name == "nb":
        if dense_embedding:
            estimator = Pipeline(
                [
                    ("scale", MinMaxScaler()),
                    ("clf", MultinomialNB()),
                ]
            )
            return estimator, {"clf__alpha": [0.1, 0.5, 1.0]}
        return MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}

    if model_name == "rf":
        return (
            RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                n_jobs=1,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
            },
        )

    if model_name == "mlp":
        return (
            MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=120,
                early_stopping=False,
                random_state=seed,
            ),
            {
                "hidden_layer_sizes": [(128,), (256, 128)],
            },
        )

    if model_name == "logreg":
        return (
            LogisticRegression(
                max_iter=1000,
                random_state=seed,
            ),
            {
                "C": [0.1, 1, 10],
            },
        )

    raise ValueError(f"Nieznany model: {model_name}")


def generate_wordclouds(dataset: TextDataset) -> list[str]:
    paths: list[str] = []
    corpus_text = " ".join(dataset.texts)
    corpus_path = PLOTS_DIR / "wordcloud_corpus.png"
    save_wordcloud(corpus_text, corpus_path)
    paths.append(str(corpus_path))

    grouped: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(dataset.texts, dataset.labels):
        grouped[label].append(text)

    for label, texts in grouped.items():
        path = PLOTS_DIR / f"wordcloud_class_{safe_filename(label)}.png"
        save_wordcloud(" ".join(texts), path)
        paths.append(str(path))

    return paths


def save_wordcloud(text: str, path: Path) -> None:
    content = text if text.strip() else "empty"
    cloud = WordCloud(width=1200, height=800, background_color="white").generate(content)
    plt.figure(figsize=(10, 6))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_document_embeddings(
    x: Any,
    labels: list[str],
    dataset_name: str,
    model_name: str,
    embedding: str,
    seed: int,
    max_points: int,
) -> list[str]:
    sampled_x, sampled_labels = sample_rows(x, labels, max_points, seed)
    paths: list[str] = []

    reductions = {
        "pca": reduce_pca,
        "tsne": reduce_tsne,
        "svd": reduce_svd,
    }

    for reduction_name, reducer in reductions.items():
        if len(sampled_labels) < 2:
            continue
        points = reducer(sampled_x, seed)
        if points is None:
            continue
        path = PLOTS_DIR / (
            f"{safe_filename(dataset_name)}_{model_name}_{embedding}_{reduction_name}_embedding.png"
        )
        scatter_plot(points, sampled_labels, path, f"{dataset_name} {embedding} {reduction_name}")
        paths.append(str(path))

    return paths


def sample_rows(
    x: Any,
    labels: list[str],
    max_points: int,
    seed: int,
) -> tuple[Any, list[str]]:
    n_rows = len(labels)
    if n_rows <= max_points:
        return x, labels

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_rows, size=max_points, replace=False))
    sampled_labels = [labels[index] for index in indices]
    return x[indices], sampled_labels


def reduce_pca(x: Any, seed: int) -> np.ndarray | None:
    dense = matrix_for_dense_reduction(x, seed)
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        return None
    return PCA(n_components=2, random_state=seed).fit_transform(dense)


def reduce_svd(x: Any, seed: int) -> np.ndarray | None:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return None
    n_components = min(2, x.shape[0] - 1, x.shape[1] - 1)
    if n_components < 2:
        return None
    return TruncatedSVD(n_components=2, random_state=seed).fit_transform(x)


def reduce_tsne(x: Any, seed: int) -> np.ndarray | None:
    dense = matrix_for_dense_reduction(x, seed, target_dimensions=50)
    if dense.shape[0] < 4 or dense.shape[1] < 2:
        return None
    perplexity = min(30, max(2, dense.shape[0] // 3))
    if perplexity >= dense.shape[0]:
        perplexity = max(1, dense.shape[0] - 1)
    return TSNE(
        n_components=2,
        random_state=seed,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
        n_jobs=1,
    ).fit_transform(dense)


def matrix_for_dense_reduction(
    x: Any,
    seed: int,
    target_dimensions: int = 100,
) -> np.ndarray:
    if sparse.issparse(x):
        if x.shape[1] > target_dimensions and min(x.shape) > 2:
            n_components = min(target_dimensions, x.shape[0] - 1, x.shape[1] - 1)
            if n_components >= 2:
                return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(x)
        return x.toarray()

    dense = np.asarray(x)
    if dense.ndim == 1:
        dense = dense.reshape(-1, 1)
    if dense.shape[1] > target_dimensions and min(dense.shape) > 2:
        n_components = min(target_dimensions, dense.shape[0] - 1, dense.shape[1] - 1)
        if n_components >= 2:
            return PCA(n_components=n_components, random_state=seed).fit_transform(dense)
    return dense


def scatter_plot(points: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    unique_labels = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(unique_labels)}
    colors = [label_to_id[label] for label in labels]

    width = 9 if len(unique_labels) <= 12 else 12
    plt.figure(figsize=(width, 7))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=colors, cmap="tab20", s=14, alpha=0.8)
    handles, _ = scatter.legend_elements()
    legend_labels = unique_labels[: len(handles)]
    if len(legend_labels) <= 20:
        plt.legend(handles, legend_labels, loc="best", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_word_embeddings(
    embedding: str,
    word_vectors: dict[str, np.ndarray],
    seed: int,
) -> list[str]:
    selected_words = select_words_for_embedding_plot(word_vectors)
    if len(selected_words) < 3:
        return []

    matrix = np.vstack([word_vectors[word] for word in selected_words])
    paths: list[str] = []

    pca_points = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    pca_path = PLOTS_DIR / f"word_embedding_{embedding}_pca.png"
    word_scatter_plot(pca_points, selected_words, pca_path, f"{embedding} PCA")
    paths.append(str(pca_path))

    if len(selected_words) >= 4:
        perplexity = min(30, max(2, len(selected_words) // 3))
        tsne_points = TSNE(
            n_components=2,
            random_state=seed,
            init="random",
            learning_rate="auto",
            perplexity=perplexity,
            n_jobs=1,
        ).fit_transform(matrix)
        tsne_path = PLOTS_DIR / f"word_embedding_{embedding}_tsne.png"
        word_scatter_plot(tsne_points, selected_words, tsne_path, f"{embedding} t-SNE")
        paths.append(str(tsne_path))

    generic_pca = PLOTS_DIR / "word_embedding_pca.png"
    generic_tsne = PLOTS_DIR / "word_embedding_tsne.png"
    word_scatter_plot(pca_points, selected_words, generic_pca, "word embeddings PCA")
    paths.append(str(generic_pca))
    if len(selected_words) >= 4:
        word_scatter_plot(tsne_points, selected_words, generic_tsne, "word embeddings t-SNE")
        paths.append(str(generic_tsne))

    return paths


def select_words_for_embedding_plot(word_vectors: dict[str, np.ndarray]) -> list[str]:
    words = [word for word in QUERY_WORDS if word in word_vectors]
    for word in list(word_vectors.keys())[:60]:
        if word not in words:
            words.append(word)
        if len(words) >= 40:
            break
    return words


def word_scatter_plot(points: np.ndarray, words: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], s=20, alpha=0.75)
    for word, point in zip(words, points):
        plt.annotate(word, xy=(point[0], point[1]), fontsize=8, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def format_similar_words(
    embedding: str,
    word_vectors: dict[str, np.ndarray],
) -> str:
    lines = [f"[{embedding}]"]
    for query in QUERY_WORDS:
        similar = most_similar(query, word_vectors, topn=5)
        if similar:
            rendered = ", ".join(f"{word} ({score:.3f})" for word, score in similar)
            lines.append(f"{query}: {rendered}")
        else:
            lines.append(f"{query}: brak słowa w słowniku")
    return "\n".join(lines)


def most_similar(
    query: str,
    word_vectors: dict[str, np.ndarray],
    topn: int,
) -> list[tuple[str, float]]:
    if query not in word_vectors:
        return []

    query_vector = word_vectors[query]
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    scores: list[tuple[str, float]] = []
    for word, vector in word_vectors.items():
        if word == query:
            continue
        denom = query_norm * np.linalg.norm(vector)
        if denom == 0:
            continue
        score = float(np.dot(query_vector, vector) / denom)
        scores.append((word, score))

    return sorted(scores, key=lambda item: item[1], reverse=True)[:topn]


def feature_importance_blocks(
    estimator: Any,
    model_name: str,
    embedding: str,
    dataset_name: str,
    feature_names: list[str],
) -> list[str]:
    fitted = unwrap_estimator(estimator)
    blocks: list[str] = []

    header = f"dataset={dataset_name} embedding={embedding} model={model_name}"

    if hasattr(fitted, "coef_"):
        classes = [str(cls) for cls in getattr(fitted, "classes_", [])]
        coef = np.asarray(fitted.coef_)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.shape[0] == 1 and len(classes) == 2:
            weights = coef[0]
            blocks.append(
                format_top_features(
                    f"{header} class={classes[1]}",
                    weights,
                    feature_names,
                )
            )
            blocks.append(
                format_top_features(
                    f"{header} class={classes[0]}",
                    -weights,
                    feature_names,
                )
            )
            return blocks
        for row_index, weights in enumerate(coef):
            class_name = classes[row_index] if row_index < len(classes) else str(row_index)
            blocks.append(
                format_top_features(
                    f"{header} class={class_name}",
                    weights,
                    feature_names,
                )
            )
        return blocks

    if hasattr(fitted, "feature_log_prob_"):
        classes = [str(cls) for cls in getattr(fitted, "classes_", [])]
        probs = np.asarray(fitted.feature_log_prob_)
        for row_index, weights in enumerate(probs):
            class_name = classes[row_index] if row_index < len(classes) else str(row_index)
            blocks.append(
                format_top_features(
                    f"{header} class={class_name}",
                    weights,
                    feature_names,
                )
            )
        return blocks

    if hasattr(fitted, "feature_importances_"):
        blocks.append(
            format_top_features(
                f"{header} global",
                np.asarray(fitted.feature_importances_),
                feature_names,
            )
        )
        return blocks

    blocks.append(f"{header}\nfeature importance niedostępne dla tego modelu.\n")
    return blocks


def unwrap_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "best_estimator_"):
        estimator = estimator.best_estimator_
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def format_top_features(
    title: str,
    weights: np.ndarray,
    feature_names: list[str],
    topn: int = 10,
) -> str:
    if len(feature_names) == 0:
        return f"{title}\nbrak feature names\n"

    limit = min(topn, len(feature_names), len(weights))
    top_indices = np.argsort(weights)[-limit:][::-1]
    lines = [title]
    for index in top_indices:
        lines.append(f"{feature_names[index]}: {weights[index]:.6f}")
    return "\n".join(lines) + "\n"


def plot_confusions(
    confusion_values: dict[tuple[str, str], dict[str, list[str]]],
    labels_order: list[str],
) -> list[str]:
    paths: list[str] = []

    for (embedding, model_name), values in confusion_values.items():
        y_true = values["true"]
        y_pred = values["pred"]
        if not y_true:
            continue

        matrix = confusion_matrix(y_true, y_pred, labels=labels_order)
        path = PLOTS_DIR / f"confusion_{embedding}_{model_name}.png"
        size = max(7, min(16, len(labels_order) * 0.7))
        fig, ax = plt.subplots(figsize=(size, size))
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels_order)
        display.plot(ax=ax, xticks_rotation=45, colorbar=False)
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))

    return paths


def write_results(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

    summary = (
        df.groupby(["dataset", "embedding", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
        )
        .sort_values(["macro_f1", "accuracy"], ascending=False)
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")


def write_feature_importance(blocks: list[str]) -> None:
    content = "\n".join(blocks).strip()
    FEATURE_IMPORTANCE_PATH.write_text(content + "\n", encoding="utf-8")


def write_similar_words(blocks: list[str]) -> None:
    content = "\n\n".join(blocks).strip()
    SIMILAR_WORDS_PATH.write_text(content + "\n", encoding="utf-8")


def _stratify_labels(labels: list[str]) -> list[str] | None:
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Dataset musi miec co najmniej dwie klasy.")
    if min(counts.values()) < 2:
        return None
    return labels


def _test_size(labels: list[str]) -> float:
    class_count = len(set(labels))
    sample_count = len(labels)
    return max(0.2, min(0.5, class_count / sample_count))


def _grid_cv(labels: list[str], seed: int) -> StratifiedKFold | None:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    splits = min(3, min(counts.values()))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_")[:80] or "unknown"
