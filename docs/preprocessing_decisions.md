## Decision 01

Remove exact duplicate rows.

## Decision 02

Keep duplicate review texts.

## Decision 03

Use 5-class classification.

## Decision 04

No empty reviews found.

## Decision 05

Most reviews are shorter than 316 words.
A Transformer model with max_length=512 is expected to cover almost all samples.

## Decision 06

Remove HTML tags.

## Decision 07

Remove URLs.

## Decision 08

Convert all text to lowercase.

## Decision 09

Tokenization analysis using bert-base-uncased shows:

- Mean token length: 148
- 95% of reviews contain fewer than 412 tokens.
- Only 3.1% of reviews exceed 512 tokens.

Therefore, max_length = 512 will be used for all Transformer-based models.







| عملیات              | Classical Models | Transformer |
| ------------------- | ---------------- | ----------- |
| Lowercase           | ✅                | ❌           |
| Remove HTML         | ✅                | ✅           |
| Remove URL          | ✅                | ✅           |
| Remove Extra Spaces | ✅                | ✅           |
| Remove Punctuation  | ✅                | ❌           |
| Remove Stopwords    | ✅                | ❌           |
| Lemmatization       | ✅                | ❌           |
| Stemming            | ❌                | ❌           |
| Remove Numbers      | ❌                | ❌           |
| Remove Emoji        | ❌                | ❌           |
