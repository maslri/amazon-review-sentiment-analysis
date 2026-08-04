عالی! 🚀

به نظر من الان دقیقاً در نقطه‌ای هستیم که از **Machine Learning کلاسیک** وارد **Deep Learning for NLP** می‌شویم.

اما همان‌طور که از اول پروژه قول داده بودم، مستقیم نمی‌رویم سراغ BERT و فقط چند خط کد بنویسیم. می‌خواهم اول واقعاً بفهمی BERT چیست، چرا ساخته شده و چرا از TF-IDF بهتر عمل می‌کند.

---

# Roadmap جدید

از اینجا به بعد:

```text
✅ 01 EDA
✅ 02 Review Length Analysis
✅ 03 Brand Analysis
✅ 04 Data Preparation
✅ 05 Dataset Split
✅ 06 TF-IDF Feature Engineering
✅ 07 TF-IDF Analysis
✅ 08 Logistic Regression
✅ 09 Linear SVM
✅ 10 Naive Bayes
✅ 11 Error Analysis
------------------------------------------------
⬜ 12 Transformer Fundamentals
⬜ 13 BERT Tokenization
⬜ 14 BERT Dataset Preparation
⬜ 15 BERT Fine-tuning
⬜ 16 BERT Evaluation
⬜ 17 Test Prediction
```

---

# چرا Notebook شماره 12؟

قبل از نوشتن حتی یک خط کد، باید جواب این سؤال‌ها را بدانی:

* Token چیست؟
* Vocabulary چیست؟
* WordPiece چیست؟
* چرا BERT جمله را می‌فهمد ولی TF-IDF نه؟
* چرا طول متن در BERT مهم است؟
* Attention چیست؟
* چرا BERT محدودیت 512 Token دارد؟

اگر این‌ها را بدانی، هنگام Fine-tuning دیگر فقط کد حفظ نمی‌کنی، بلکه می‌فهمی چه اتفاقی در حال رخ دادن است.

---

# Notebook جدید

```
12_transformer_fundamentals.ipynb
```

---

# Cell 1 — Why Move Beyond TF-IDF?

Markdown:

```markdown
# Why do we need Transformer models?

Traditional machine learning models represent text as a bag of words.
They ignore word order and contextual meaning.

Transformer-based models such as BERT generate contextual embeddings,
allowing the same word to have different meanings depending on its context.
```

بعد یک مثال ساده:

```
I bought an Apple phone.

Apple released a new MacBook.
```

TF-IDF فقط کلمه **Apple** را می‌بیند.

BERT متوجه می‌شود هر دو درباره شرکت Apple هستند.

و اگر بنویسیم:

```
I ate an apple.
```

بردار این "apple" کاملاً متفاوت خواهد بود.

---

# Cell 2 — What is a Token?

اینجا فقط متن و چند مثال.

مثلاً:

```
This movie is fantastic!
```

Tokenها:

```
This
movie
is
fantastic
!
```

اما بعد مثال سخت‌تر:

```
unbelievable
```

ممکن است تبدیل شود به:

```
un
##believable
```

---

# Cell 3 — WordPiece

این مهم‌ترین مفهوم قبل از BERT است.

مثلاً:

```
playing
```

به

```
play
##ing
```

تبدیل می‌شود.

---

یا

```
unhappiness
```

به

```
un
##happy
##ness
```

---

اینجا توضیح می‌دهیم چرا BERT می‌تواند کلمات جدید را هم پردازش کند.

---

# Cell 4 — Special Tokens

معرفی:

```
[CLS]
```

```
[SEP]
```

```
[PAD]
```

```
[MASK]
```

با مثال:

```
[CLS]
This movie is amazing
[SEP]
```

---

# Cell 5 — Why Maximum Length Exists

اینجا دقیقاً از نتایج پروژه خودمان استفاده می‌کنیم.

یادت هست؟

Review Length را بررسی کردیم.

دیدیم:

```
90%

≈1200 characters
```

---

حالا توضیح می‌دهیم:

BERT با Character کار نمی‌کند.

با

```
Token
```

کار می‌کند.

تقریباً:

```
1 token

≈ 0.75 word
```

---

بعد خواهیم دید

اکثر Reviewهای ما

کمتر از

```
256 Token
```

هستند.

---

# Cell 6 — Attention

فقط مفهوم.

بدون فرمول.

مثال:

```
The phone is great but the battery is terrible.
```

وقتی BERT کلمه

```
terrible
```

را می‌خواند،

می‌تواند به

```
battery
```

توجه کند.

نه فقط کلمه قبلی.

این همان چیزی است که TF-IDF نمی‌تواند انجام دهد.

---

# Cell 7 — Why BERT Should Improve Our Results

Markdown:

```
Current models fail mostly on:

4 ↔ 5

3 ↔ 4

2 ↔ 3
```

چون جمله‌ها

مبهم‌اند.

مثلاً:

```
The product is good,
but the battery life is disappointing.
```

TF-IDF

کلمات

```
good

disappointing
```

را جداگانه می‌بیند.

BERT

کل جمله را می‌فهمد.

---

# Cell 8 — Summary

Markdown:

```
Transformer models solve several limitations of classical NLP:

✓ Context awareness
✓ Word order
✓ Long-range dependencies
✓ Better semantic understanding
```

---

# چرا این Notebook را می‌سازیم؟

چون وقتی ۶ ماه بعد به پروژه برگردی، اگر مستقیم از `transformers` استفاده کرده باشی، احتمالاً فقط کد را می‌بینی.

اما اگر این Notebook را داشته باشی، همیشه یادت می‌آید که **چرا** از BERT استفاده کردی، نه فقط **چگونه**.

---

# بعد از این Notebook

برای اولین بار وارد کدنویسی با کتابخانه Hugging Face می‌شویم:

```
13_bert_tokenization.ipynb
```

و آنجا:

* اولین `AutoTokenizer` را می‌سازیم.
* اولین Tokenهای واقعی دیتاست Amazon را می‌بینیم.
* تعداد Tokenهای Reviewها را اندازه می‌گیریم.
* بر اساس داده‌های خودمان، مقدار مناسب `max_length` را انتخاب می‌کنیم.

این اولین قدم عملی در دنیای Transformerها خواهد بود و از آن به بعد وارد Fine-tuning مدل BERT روی دیتاست خودمان می‌شویم.
