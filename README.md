# Restaurant Review Sentiment Analysis

### From raw reviews to text features and sentiment predictions

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLP-NLTK-154F3C)
![scikit-learn](https://img.shields.io/badge/Models-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

A hands-on natural language processing project for classifying English restaurant reviews as **positive** or **negative**. Explore text cleaning, stopword removal, stemming, Bag-of-Words features, logistic regression, and random forest classification. A companion notebook explains binary word presence, n-grams, and TF-IDF with small examples.

**[Quick start](#quick-start) · [Notebooks](#notebooks) · [Dataset](#dataset) · [Evaluation](#evaluation) · [Preparation notes](docs/preparation-notes.md)**

## What you will learn

- Turn raw review text into a consistent representation for training and prediction.
- Preserve negation words such as **not**, **no**, and **never** during cleaning.
- Build a vocabulary using training reviews and apply it to held-out reviews.
- Compare two traditional classifiers on the same held-out partition.
- Interpret accuracy, precision, recall, F1, ROC AUC, and confusion matrices.
- Understand how word counts differ from TF-IDF weights.

## Quick start

Use **Python 3.11 or later**. The commands below work from macOS Terminal or the VS Code terminal.

### 1. Get the project

Clone the repository:

```bash
git clone https://github.com/KingaTshering10/Restaurant-review-sentiment-analysis.git restaurant-review-sentiment-analysis
cd restaurant-review-sentiment-analysis
```

Alternatively, extract the project ZIP and open the extracted folder in VS Code.

### 2. Create the environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

The final command installs the language data used by NLTK. It requires an internet connection the first time. After the packages and stopwords are installed, both notebooks can run with the included local dataset. See [NLTK's data installation guide](https://www.nltk.org/data.html) if the resource cannot be found.

### 3. Launch the notebooks

```bash
python -m jupyterlab
```

Open a notebook and run its cells from top to bottom. In VS Code, you can open the notebook directly and select the `.venv` interpreter as its kernel. Start from the project folder so the relative dataset path resolves.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `bow_tfidf_basics.ipynb` | Short examples of binary Bag-of-Words, count-based n-grams, and TF-IDF |
| `restaurant_sentiment_analysis.ipynb` | Dataset inspection, text cleaning, classifier training, evaluation, and new-review predictions |

Start with the feature tutorial if you are new to NLP, then continue with the sentiment notebook. The classifier uses **unigram Bag-of-Words**; TF-IDF and n-grams are illustrated separately rather than evaluated as alternative classifiers here.

## Dataset

`Restaurant_Reviews.csv` is included, so no separate dataset download is needed.

| Property | Supplied CSV |
| --- | ---: |
| Reviews | 1,000 |
| Positive labels (`Liked = 1`) | 500 |
| Negative labels (`Liked = 0`) | 500 |
| Missing values | 0 |
| Exact duplicate reviews | 4 |
| Reviews after removing exact duplicates | 996 |

Each row has two columns: **`Review`**, containing the text, and **`Liked`**, containing its 0/1 label. The source CSV is preserved unchanged; duplicate removal happens in memory during the notebook run.

The supplied data closely matches the Yelp subset of [UCI's Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), attributed to Dimitrios Kotzias (2015), [DOI: 10.24432/C57604](https://doi.org/10.24432/C57604). Comparison with the official download found 995 matching rows after whitespace trimming, with five text differences. The [mirror recorded in the original notebook](https://raw.githubusercontent.com/ubaid-shah/datasets/main/Restaurant_Reviews.tsv) is also retained for provenance. UCI lists its upstream dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); credit remains with the original providers.

`docs/original_corpus_dataset.csv` preserves the previously exported corpus. It uses the original cleaning rules and is **not an input to the corrected training workflow**.

## How the classifier works

1. Load the dataset, validate its columns and labels, and remove exact duplicate reviews.
2. Lowercase text, normalize common negative contractions, remove non-letter characters, remove stopwords while keeping negations, and apply Porter stemming.
3. Build one approximately 80/20 holdout using the first split of a shuffled five-fold `StratifiedGroupKFold`, with seed `0`. Reviews that become identical after cleaning stay in the same partition.
4. Fit `CountVectorizer(max_features=1500)` on training text only; transform test text using that fitted vocabulary.
5. Train logistic regression with up to 1,000 iterations and a random forest with 100 trees. Both use seed `0`.
6. Evaluate each model, then apply the same cleaner and vectorizer to new reviews.

The notebook uses sparse feature matrices. It does not fit its vocabulary to the complete dataset before splitting; this follows [scikit-learn's guidance on preventing preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Evaluation

The notebook prints per-class classification reports, a comparison table, and a confusion matrix for each classifier. Precision, recall, and F1 in the comparison table refer to the **positive** class.

A verified local run used **797 training reviews**, **199 test reviews** (99 negative / 100 positive), and **1400 learned features**. No cleaned review string appeared in both partitions.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 77.39% | 0.7835 | 0.7600 | 0.7716 | 0.8757 |
| Random forest | 84.92% | 0.9070 | 0.7800 | 0.8387 | 0.8949 |

Validation and environment details are recorded in [preparation notes](docs/preparation-notes.md). Results from this small holdout describe one run, and can change with dependency versions or partition choices. The original notebook's saved scores use a different preprocessing and splitting protocol and are documented separately rather than treated as directly comparable results.

## Try your own review

Run the main notebook first. In its final cell, change:

```python
user_review = "The food was delicious and the staff were friendly."
```

Run the cell again to see a label from each classifier. You can also use the notebook's fitted objects:

```python
cleaned = preprocess_review(user_review)
features = cv.transform([cleaned])
label = logistic_model.predict(features)[0]
print("Positive" if label == 1 else "Negative")
```

The final notebook cell rejects inputs that clean to an empty string or have no words in the training vocabulary. Keeping a word such as “not” does not guarantee correct handling of negation: these unigram models can still misclassify phrases, sarcasm, and mixed sentiment.

## Project files

| File | Purpose |
| --- | --- |
| `README.md` | Overview and setup guide |
| `requirements.txt` | Python dependencies and notebook tools |
| `bow_tfidf_basics.ipynb` | Text-feature tutorial |
| `restaurant_sentiment_analysis.ipynb` | Sentiment analysis workflow |
| `Restaurant_Reviews.csv` | Original review text and labels |
| `docs/original_corpus_dataset.csv` | Archived corpus from the supplied project |
| `docs/preparation-notes.md` | Corrections, historical outputs, and verification details |
| `.gitignore` | Excludes environments, notebook checkpoints, caches, and generated files |

Running the main notebook writes the newly cleaned corpus to **`artifacts/cleaned_reviews.csv`**, without an index column. Generated files are excluded from Git. Models remain in the notebook session; no pretrained weights or web application are included.

## Limitations and possible extensions

This is an educational project on a small, balanced English-language dataset. The holdout is grouped by identical cleaned text, while other near-duplicates are not detected. Both classifiers are inspected on the same holdout; model selection and hyperparameter tuning need a separate validation procedure. Unigram features capture word associations and do not represent word order, complete negation scope, or restaurant aspects such as food versus service.

Possible extensions include evaluating TF-IDF and bigrams, cross-validation with text grouping, error analysis, and comparison with pretrained language models. None of those extensions are claimed as completed experiments here.

## Acknowledgments

Maintained by **[Kinga Tshering](https://github.com/KingaTshering10)**. Prepared from the supplied NLP training materials, retaining both learning workflows and the original data. See [preparation notes](docs/preparation-notes.md) for the changes. No project code license was included in the supplied archive.
