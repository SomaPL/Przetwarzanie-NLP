# Przetwarzanie NLP - Laboratorium 1/2

Bot Telegram do przetwarzania, klasyfikacji pojedynczych wiadomości oraz eksperymentów klasyfikacji tekstu na datasetach.

## Wymagania
- Python 3.11 albo 3.12
- Telegram bot token z BotFather

Uwaga: obecne zależności projektu, szczególnie spaCy, nie działają poprawnie na Pythonie 3.14.

## Instalacja

Jeśli masz już `.venv` utworzone na innej wersji Pythona, usuń je przed instalacją:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
```

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download pl_core_news_sm
```

Utwórz plik `.env` w katalogu projektu:

```env
BOT_TOKEN=twoj_token_z_botfather
```

## Uruchomienie

```powershell
.\.venv\Scripts\Activate.ps1
python bot.py
```

## Laboratorium 2

Bot obsługuje komendę:

```text
/classify dataset=<dataset_name> method=<model> gridsearch=<true/false> run=<n>
```

Przykłady:

```text
/classify dataset=20news_group method=all gridsearch=false run=1
/classify dataset=amazon method=logreg gridsearch=true run=2
/classify dataset=imdb method=rf,nb gridsearch=false run=3
/classify dataset=ag_news method=rf,nb gridsearch=false run=3
```

Dodatkowo można ograniczyć embeddingi, co przydaje się do szybkich testów:

```text
/classify dataset=demo method=logreg gridsearch=false run=1 embedding=tfidf
```

Obsługiwane modele:

```text
nb, rf, mlp, logreg, all
```

Bot uruchamia reprezentacje:

```text
bow, tfidf, word2vec, glove
```

Dataset `20news_group` jest pobierany przez `sklearn.datasets.fetch_20newsgroups`. Datasety `imdb`, `amazon` i `ag_news` wrzuć jako CSV do katalogu `data/`, np. `data/imdb.csv`, `data/amazon.csv`, `data/ag_news.csv`. Plik powinien mieć kolumny typu `text,label` albo `review,sentiment`.

Jeśli chcesz użyć prawdziwego pretrained GloVe, dodaj plik:

```text
data/glove.6B.50d.txt
```

Bez tego pliku bot użyje lokalnej reprezentacji opartej na log-zliczeniach i SVD.

Wyniki są zapisywane do:

```text
lab2results.csv
lab2results_summary.csv
lab2_feature_importance.txt
lab2_similar_words.txt
lab2plots/
```
