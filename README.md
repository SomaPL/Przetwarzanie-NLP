# Przetwarzanie NLP - Laboratorium 1/2/3

Bot Telegram do przetwarzania, klasyfikacji pojedynczych wiadomości, eksperymentów klasyfikacji tekstu oraz analizy sentymentu.

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
python -c "import stanza; stanza.download('pl')"
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

## Laboratorium 3

Główny dataset dla Lab 3 jest po polsku i znajduje się w pliku:

```text
sentiment_dataset.csv
```

Format:

```csv
text,label
"Uwielbiam ten film",pozytywny
"To był zwykły dzień",neutralny
"Ten produkt jest fatalny",negatywny
```

Nowe komendy:

```text
/sentiment method=<metoda> text="tekst"
/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>
/compare dataset=<amazon|imdb|custom> methods=<lista_metod>
/add_sentiment "tekst" "etykieta"
/models
/help
```

Przykłady:

```text
/sentiment method=rule text="To był naprawdę świetny film"
/sentiment method=nb dataset=custom text="Produkt przyszedł uszkodzony"
/train model=lstm dataset=custom epochs=10 max_len=200
/compare dataset=custom methods=rule,nb,rf,textblob
/add_sentiment "Obsługa była poprawna, ale niczym mnie nie zachwyciła" "neutralny"
```

Komenda `/add_sentiment` zapisuje cały podany tekst jako jeden rekord. Jeżeli tekst ma kilka zdań, nie jest dzielony automatycznie na osobne przykłady.

Metody dostępne w `/sentiment`:

```text
rule, nb, rf, transformer, textblob, stanza, simplernn, lstm, gru
```

Modele sekwencyjne `SimpleRNN`, `LSTM` i `GRU` trzeba najpierw wytrenować komendą `/train`. Model jest zapisywany jako `.h5`, a tokenizer i encoder etykiet są zapisywane obok niego w katalogu `models/`, np.:

```text
models/lstm_custom.h5
models/lstm_custom_tokenizer.h5
models/lstm_custom_label_encoder.h5
models/lstm_custom_meta.json
```

Podczas predykcji `/sentiment method=lstm ...` bot wczytuje zapisany model z pliku, zamiast trenować go od nowa.

Wyniki i wykresy Lab 3:

```text
lab3results.csv
lab3plots/train_history_lstm_custom.png
lab3plots/confusion_lstm_custom.png
lab3plots/compare_methods_custom.png
lab3plots/wordcloud_pozytywny.png
lab3plots/class_distribution_custom.png
```

Parametry treningu można zmieniać w komendzie:

```text
/train model=gru dataset=custom epochs=15 max_len=150 batch_size=32
```

Proponowane eksperymenty z długością sekwencji:

```text
max_len=80
max_len=120
max_len=200
max_len=300
```

Dla każdego wariantu można uruchomić ten sam model i porównać `accuracy`, `macro_f1` oraz wykres historii uczenia. Większe `max_len` zachowuje więcej słów z tekstu, ale zwiększa czas treningu.

Uwaga: `transformer` przy pierwszym użyciu pobiera model `cardiffnlp/twitter-xlm-roberta-base-sentiment`. `stanza` wymaga pobranego modelu języka polskiego komendą z sekcji instalacji.

## Przygotowanie do Laboratorium 4

Lab 4 będzie korzystał głównie z internetowego linkowania encji:

```text
Wikidata API / Wikidata SPARQL
Wikipedia API
Hugging Face models
Ollama lokalnie przez http://localhost:11434
```

Lokalna baza wiedzy może zostać dodana tylko jako zapasowe źródło, ale główna ścieżka będzie online.

Dodatkowe biblioteki dopisane do `requirements.txt`:

```text
langdetect
networkx
sentencepiece
sacremoses
wikipedia-api
SPARQLWrapper
```

Po aktualizacji zależności uruchom:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ollama musi być uruchomiona lokalnie, a wybrany model powinien odpowiadać w terminalu przez:

```powershell
ollama run nazwa_modelu
```

## Laboratorium 4

Nowe komendy:

```text
/ner method=<spacy|stanza> text="tekst"
/nel text="tekst" language=<en|pl>
/ned entity="nazwa" context="tekst kontekstu" language=<en|pl>
/translate text="tekst" target_lang=<en|pl|de|fr|es>
/summarize text="tekst" summary_type=<extractive|abstractive|bullets> length=<short|medium|long>
/analyze_entities text="tekst" link=<true|false>
/knowledge_graph text="tekst"
/language_detect text="tekst"
```

Przykłady:

```text
/ner method=spacy text="Steve Jobs, założyciel Apple'a, urodził się w San Francisco."
/ner method=stanza text="Elon Musk posiada firmę Tesla oraz xAI w Austin."
/nel text="Steve Jobs" language=en
/ned entity="Apple" context="Steve Jobs założył Apple w Kalifornii" language=en
/translate text="The quick brown fox jumps over the lazy dog" target_lang=pl
/summarize text="Tutaj wklej dłuższy tekst" summary_type=abstractive length=medium
/summarize text="Tutaj wklej dłuższy tekst" summary_type=bullets length=short
/analyze_entities text="Elon Musk posiada firmę Tesla oraz xAI w Austin." link=true
/knowledge_graph text="Elon Musk posiada firmę Tesla oraz xAI w Austin."
/language_detect text="To jest przykładowy tekst"
```

NER działa przez `spaCy` albo `Stanza`. NEL/NED korzysta z Wikidata i Wikipedii przez internet. Tłumaczenie korzysta z modeli Helsinki-NLP/Opus-MT przez `transformers`. Dla par bez bezpośredniego modelu, np. `pl -> de`, bot tłumaczy przez angielski: `pl -> en -> de`.

Podsumowania są generowane przez lokalne API Ollama:

```text
http://localhost:11434/api/generate
```

Domyślny model można zmienić zmienną środowiskową:

```powershell
$env:OLLAMA_MODEL="nazwa_modelu"
```

Wyniki Lab 4 są zapisywane do:

```text
lab4results/lab4_entities.csv
lab4results/lab4_entity_links.csv
lab4results/lab4_summaries.csv
lab4results/lab4_translations.csv
lab4plots/knowledge_graph_<timestamp>.png
```
