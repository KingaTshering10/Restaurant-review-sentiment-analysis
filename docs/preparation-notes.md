# Preparation notes

The supplied archive contains two notebooks, a 1,000-row restaurant review dataset, an exported cleaned corpus, and a requirements file listing only NLTK.

## Corrections in this repository copy

- Describe the main task as restaurant-review **sentiment analysis**; the original introduction described hotels, while its filename used “semantic analysis.”
- Remove four exact duplicate reviews before splitting. The working dataset has 996 reviews: 499 positive and 497 negative.
- Retain the original 1,000-row CSV unchanged. Keep the original derived corpus in `docs/original_corpus_dataset.csv` for reference; it is not used to train the corrected models.
- Preserve `no`, `nor`, `not`, and `never` during stopword removal. Normalize common negative contractions before removing punctuation. This retains useful tokens but does not implement full negation-scope understanding.
- Use one cleaning function for training reviews and new inputs.
- Replace the hard-coded loop over 1,000 rows with iteration over the actual dataframe.
- Split the cleaned text before fitting the vectorizer. Fit its vocabulary on training text only and transform test text with the same fitted vectorizer.
- Use the first holdout from a shuffled five-fold `StratifiedGroupKFold` with `random_state=0`, targeting an approximately 80/20 split. Group identical cleaned text to keep normalization-equivalent reviews in one partition.
- Keep the original Bag-of-Words classification approach with up to 1,500 unigram features. TF-IDF and n-grams remain separate tutorial demonstrations.
- Keep logistic regression and random forest as distinct variables so fitting the second model does not overwrite the first. Configure their random seeds and allow up to 1,000 logistic-regression iterations.
- Report per-model accuracy, positive-class precision, recall, F1, ROC AUC, and confusion matrices. Apply the same fitted preprocessing to sample predictions.
- Remove blanket warning suppression and clear the old saved outputs.
- Complete the dependency list and add instructions for installing the NLTK stopwords resource.
- Exclude macOS metadata and notebook checkpoints from the repository.

## Historical outputs

The original notebook records logistic-regression accuracy **0.710** and random-forest accuracy **0.685** on its original, unstratified 200-review split. Its recorded random-forest confusion matrix is `[[84, 13], [50, 53]]`, with rows and columns ordered negative, positive.

These are historical outputs, not results for the corrected protocol. The original vocabulary was fitted before the split, stopwords removed negations, and sample predictions skipped training-time cleaning. The original examples “not nice service” and “not a great taste” were both labeled positive.

The four exact duplicate rows are removed before the new split. Reviews that become identical after cleaning are grouped into one partition. Other near-duplicate or differently worded reviews are not grouped; this small holdout is a learning exercise rather than a robust benchmark. Both models are evaluated on the same holdout for demonstration, so selecting or tuning a model from these results requires a separate validation procedure.

The original exported corpus has an index column and one empty cleaned string, which pandas reads as a missing value by default. It remains an archived reference. Running the corrected notebook exports a newly generated corpus under `artifacts/`.

No project license or reliable original training-notebook authorship metadata was supplied. The source data link recorded in the original notebook is retained in the README; ownership is not assigned to the repository maintainer.


## Verified run — 26 October 2026

Both notebooks were executed from top to bottom. The final corrected sentiment workflow trained both classifiers and generated the comparison metrics, plots, sample predictions, and cleaned-corpus export. The prepared notebook files retain cleared outputs; the values below document the verification run.

- Training / test rows: 797 / 199.
- Positive training / test labels: 399 / 100.
- Learned vocabulary: 1400 features.
- Identical cleaned text shared between partitions: 0.
- Environment: nltk 3.9.2, numpy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 77.39% | 0.7835 | 0.7600 | 0.7716 | 0.8757 |
| Random forest | 84.92% | 0.9070 | 0.7800 | 0.8387 | 0.8949 |

Confusion matrices use rows = actual and columns = predicted, ordered negative then positive:

- Logistic regression: `[[78, 21], [24, 76]]`.
- Random forest: `[[91, 8], [22, 78]]`.

Checks also confirmed that negations survive cleaning, the learned vocabulary comes only from training text, and predictions use the same cleaning function. Both models still predict “not a great taste” as positive, illustrating the limits of unigram features. No claim of general negation understanding is made.
