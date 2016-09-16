# kodealltag-spelling-corrector

## Preprocessing
* remove comments (heuristic on line starting with `">"`)
* sentence splitting with `NLTK's SentTokenizer (german model)`

## Main Spelling Correction
* simple regex pretokenizer: `"[\w|�]+|[^\w\s]+"`
* tokens are then processed by `BadCharReplacer` (modified SpellingCorrection with EditDistance and GoogleBooks Unigram Corpus)

## Tagging/Lemmatizing
* the cleaned tokens are then processed by TreeTagger (Python-Wrapper)
* written to same location as the respective original files with `*.tok` ending
