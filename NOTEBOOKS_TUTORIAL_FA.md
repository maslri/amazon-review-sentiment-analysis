# راهنمای آموزشی کامل Notebookهای پروژه پیش‌بینی امتیاز Amazon Reviews

این سند توضیح قدم‌به‌قدم کارهایی است که در Notebookهای پروژه انجام شده است. هدف فقط توضیح «چه کدی نوشته شده» نیست؛ در هر بخش توضیح می‌دهیم چرا آن تصمیم گرفته شده، چه مسئله‌ای را حل می‌کند، چه جایگزین‌هایی وجود داشت و هنگام اجرای پروژه باید به چه نکاتی توجه کرد.

## 1. تصویر کلی پروژه

مسئله، یک دسته‌بندی چندکلاسه متنی است. مدل باید با دیدن اطلاعات یک review، امتیاز آن را از میان پنج کلاس زیر پیش‌بینی کند:

```text
1, 2, 3, 4, 5
```

ستون هدف در داده آموزشی `overall` نام دارد. داده test این ستون را ندارد و قرار است در پایان فایلی به نام `q2_submission.csv` با یک ستون `predicted` برای آن ساخته شود.

جریان فعلی پروژه چنین است:

```text
داده خام
→ شناخت داده با EDA
→ پاک‌سازی سبک متن
→ ساخت داده متوازن
→ جداسازی train و validation
→ آموزش RoBERTa + LoRA
→ آموزش DeBERTa-v3 + LoRA
→ مقایسه مدل‌ها
→ انتخاب مدل نهایی
→ پیش‌بینی test
→ ساخت submission
```

Notebookهای فعلی:

```text
01_eda.ipynb
02_preprocessing.ipynb
02_preprocessing_kaggle.ipynb
03_roberta_keras.ipynb
04_deberta_v3_keras.ipynb
```

نسخه محلی preprocessing از کد موجود در `src/` استفاده می‌کند. نسخه Kaggle مستقل است تا بدون نیاز به آپلود کل repository اجرا شود.

---

# 2. معیارهای ارزیابی

## Accuracy و Micro-F1

در دسته‌بندی تک‌برچسبی چندکلاسه، هر نمونه فقط یک کلاس واقعی و یک کلاس پیش‌بینی‌شده دارد. در این وضعیت معمولاً Micro-F1 با Accuracy برابر است:

```text
Micro-F1 = تعداد پیش‌بینی‌های درست / تعداد کل نمونه‌ها
```

معیار رسمی پروژه Micro-F1 است؛ بنابراین انتخاب نهایی مدل باید عمدتاً بر اساس validation micro-F1 انجام شود.

## Macro-F1

Macro-F1 ابتدا F1 هر کلاس را جداگانه محاسبه و سپس میانگین ساده آن‌ها را می‌گیرد:

```text
Macro-F1 = (F1 کلاس 1 + ... + F1 کلاس 5) / 5
```

در این معیار، کلاس کم‌تعداد و پرتعداد وزن مساوی دارند. به همین دلیل Macro-F1 برای تشخیص ضعف مدل در کلاس‌های دشوار 2 و 3 مناسب است.

## Weighted-F1

Weighted-F1 میانگین F1 کلاس‌ها با وزن تعداد نمونه‌های هر کلاس است. این معیار بین Micro-F1 و Macro-F1 قرار می‌گیرد و رفتار مدل روی توزیع واقعی داده را بهتر نشان می‌دهد.

## چرا نتایج validation متوازن را نباید مستقیماً با validation طبیعی مقایسه کرد؟

در داده خام، کلاس 5 بسیار بیشتر از کلاس‌های دیگر است. مدلی که کلاس 5 را خوب پیش‌بینی کند می‌تواند Micro-F1 بالایی بگیرد، حتی اگر روی کلاس‌های 2 و 3 ضعیف باشد. در validation متوازن هر کلاس تعداد یکسانی دارد؛ بنابراین Micro-F1 دیگر از فراوانی کلاس 5 سود نمی‌برد.

برای نمونه، این دو نتیجه لزوماً نشان نمی‌دهند مدل اول بهتر است:

```text
Micro-F1 = 0.75 روی validation نامتوازن
Micro-F1 = 0.63 روی validation متوازن
```

برای مقایسه منصفانه، مدل‌ها باید روی دقیقاً یک validation مشترک ارزیابی شوند.

---

# 3. Notebook شماره 01: تحلیل اکتشافی داده

فایل: `notebooks/01_eda.ipynb`

هدف EDA شناخت داده قبل از هرگونه تغییر است. این Notebook چیزی را آموزش نمی‌دهد و فایل پردازش‌شده تولید نمی‌کند.

## 3.1 تعیین مسیرها

کد ابتدا تشخیص می‌دهد اجرا از root پروژه انجام شده یا داخل پوشه `notebooks`:

```python
PROJECT_ROOT = (
    Path.cwd().resolve().parent
    if Path.cwd().name == "notebooks"
    else Path.cwd().resolve()
)
```

این کار باعث می‌شود Notebook در هر دو حالت مسیرهای صحیح را پیدا کند.

سه فایل خام مشخص می‌شوند:

```text
train_data.csv
test_data.csv
title_brand.csv
```

- `train_data.csv`: reviewها به همراه `overall`
- `test_data.csv`: reviewهای بدون target
- `title_brand.csv`: اطلاعات محصول بر اساس `asin`

پارامتر `low_memory=False` در `read_csv` باعث می‌شود pandas هنگام حدس‌زدن نوع ستون‌ها، فایل را به قطعات کوچک با حدس‌های ناسازگار تقسیم نکند.

## 3.2 بررسی شکل و schema

برای هر دیتاست تعداد سطر و ستون نمایش داده می‌شود. سپس چند ردیف اول و نام ستون‌ها چاپ می‌شود.

این مرحله چند سؤال را پاسخ می‌دهد:

- آیا فایل درست بارگذاری شده است؟
- آیا test واقعاً فاقد `overall` است؟
- آیا `asin` برای merge وجود دارد؟
- آیا نام ستون‌ها با انتظار کد یکسان است؟

## 3.3 بررسی توزیع target

با `value_counts()` تعداد هر امتیاز محاسبه می‌شود. درصد هر کلاس نیز نمایش داده می‌شود.

نمودار میله‌ای نشان می‌دهد داده خام به‌شدت به سمت امتیاز 5 متمایل است. این عدم توازن روی تفسیر Accuracy و Micro-F1 اثر مستقیم دارد.

## 3.4 مقادیر missing و duplicate

برای هر ستون دو مقدار محاسبه می‌شود:

```text
missing_count
missing_percentage
```

همچنین سه وضعیت جدا بررسی می‌شود:

1. ردیف کاملاً تکراری
2. متن review تکراری
3. `reviewText` خالی

ردیف کاملاً تکراری باید پیش از split حذف شود؛ زیرا اگر یک نسخه وارد train و نسخه دیگر وارد validation شود، مدل عملاً پاسخ validation را قبلاً دیده است.

متن تکراری به‌تنهایی حذف نشده است، چون ممکن است دو رکورد با متن مشابه اما مشخصات واقعی متفاوت وجود داشته باشند. این موضوع در صورت نیاز می‌تواند در تحلیل leakage آینده سخت‌گیرانه‌تر بررسی شود.

## 3.5 تحلیل طول review

طول متن بر حسب تعداد کاراکتر محاسبه می‌شود:

```python
review_length = review_text.str.len()
```

علاوه بر میانگین و میانه، صدک‌های 90، 95، 99، 99.5 و 99.9 بررسی می‌شوند. دلیل استفاده از صدک این است که توزیع طول review راست‌چوله است؛ چند متن بسیار بلند می‌توانند میانگین را گمراه‌کننده کنند.

دو histogram نمایش داده می‌شود:

- کل توزیع
- توزیع تا صدک 99

نمودار دوم شکل بخش اصلی داده را بدون تأثیر reviewهای بسیار طولانی نشان می‌دهد.

## 3.6 کنترل کیفیت متن

وجود موارد زیر بررسی می‌شود:

```text
متن خالی
HTML
URL
line break
```

این تحلیل مبنای preprocessing است. HTML و URL معمولاً سیگنال مفیدی برای rating نیستند و فضای token را مصرف می‌کنند. line break با فاصله استاندارد جایگزین می‌شود.

## 3.7 پوشش اطلاعات محصول

فایل محصول بر اساس `asin` deduplicate و سپس coverage آن در reviewها محاسبه می‌شود. `asin` شناسه محصول و کلید ارتباط دو جدول است.

همچنین برندهای پرتکرار و میانگین امتیاز آن‌ها بررسی می‌شود. با این حال title و brand به ورودی نهایی مدل اضافه نشدند، چون آزمایش ablation قبلی بهبود validation نشان نداد.

## 3.8 نتیجه EDA

تصمیم‌های اصلی حاصل از EDA:

- مسئله پنج‌کلاسه است.
- target خام نامتوازن است.
- exact duplicate باید حذف شود.
- متن‌های خالی نباید وارد train شوند.
- preprocessing مدل Transformer باید سبک باشد.
- test نباید برای انتخاب مدل استفاده شود.

---

# 4. Notebook شماره 02: preprocessing محلی

فایل: `notebooks/02_preprocessing.ipynb`

این Notebook دیتاست پردازش‌شده اصلی را می‌سازد.

## 4.1 تنظیمات مهم

```python
SAMPLES_PER_CLASS = 50_000
VALIDATION_PER_CLASS = 2_000
RANDOM_STATE = 42
SUMMARY_MAX_WORDS = 40
```

`SAMPLES_PER_CLASS` کل تعداد انتخاب‌شده از هر کلاس است. بنابراین حجم کل داده:

```text
50,000 × 5 = 250,000
```

از هر کلاس 2,000 نمونه برای validation کنار گذاشته می‌شود:

```text
Train هر کلاس:      48,000
Validation هر کلاس:  2,000
Train کل:           240,000
Validation کل:       10,000
```

validation ثابت انتخاب شده تا با افزایش حجم آزمایش، داده جدید به train اضافه شود و validation بی‌دلیل بزرگ نشود.

`RANDOM_STATE=42` نمونه‌گیری را reproducible می‌کند؛ یعنی اجرای مجدد با داده و کد یکسان، همان split را می‌سازد.

## 4.2 تابع پاک‌سازی Transformer

نسخه محلی تابع زیر را از `src.preprocessing.cleaner` وارد می‌کند:

```python
preprocess_for_transformer
```

این تابع:

- HTML را حذف می‌کند.
- URL را حذف می‌کند.
- whitespaceهای پشت سرهم را یکی می‌کند.
- ورودی غیررشته‌ای و missing را امن مدیریت می‌کند.

اما موارد زیر را حفظ می‌کند:

- حروف بزرگ و کوچک
- punctuation
- stopwordها
- negation مانند `not`
- عددها
- emoji
- ساختار طبیعی جمله

مدل‌های BERT-family روی متن طبیعی pretrain شده‌اند. پاک‌سازی تهاجمی مانند stemming یا حذف `not` می‌تواند اطلاعات احساسی را از بین ببرد.

## 4.3 حذف duplicate و متن نامعتبر

ابتدا exact duplicate حذف می‌شود:

```python
deduplicated_df = raw_df.drop_duplicates()
```

سپس review قبل از نمونه‌گیری به‌صورت آزمایشی پاک می‌شود. اگر نتیجه خالی باشد، آن ردیف eligible نیست:

```python
cleaned_review_candidate = ...map(preprocess_for_transformer)
eligible_review_mask = cleaned_review_candidate.str.strip().ne("")
```

این ترتیب مهم است. اگر ابتدا 50 هزار نمونه انتخاب و بعد متن خالی حذف شود، تعداد کلاس‌ها دیگر برابر نخواهد بود. با حذف متن خالی قبل از sampling، pandas به‌جای آن یک review معتبر دیگر از همان کلاس انتخاب می‌کند.

این فیلتر فقط روی train انجام می‌شود. هیچ ردیفی از test حذف نمی‌شود، چون ترتیب submission باید دقیقاً با test اصلی یکسان بماند.

## 4.4 ساخت validation ثابت

ابتدا validation مستقیماً از تمام داده eligible انتخاب می‌شود:

```python
validation_raw = deduplicated_df.groupby("overall").sample(
    n=VALIDATION_PER_CLASS,
    replace=False,
    random_state=RANDOM_STATE,
)
```

بعد indexهای validation از training pool حذف می‌شوند:

```python
training_pool = deduplicated_df.drop(index=validation_raw.index)
```

در نهایت train از pool باقی‌مانده انتخاب می‌شود. این طراحی دو مزیت دارد:

1. train و validation هم‌پوشانی ندارند.
2. اگر `SAMPLES_PER_CLASS` تغییر کند، validation ثابت می‌ماند.

`replace=False` یعنی هیچ نمونه‌ای با تکرار انتخاب نمی‌شود. oversampling با تکرار می‌توانست باعث حفظ‌کردن reviewها و overfitting شود.

ستون موقت `_split` مشخص می‌کند هر ردیف متعلق به train است یا validation. این ستون پیش از ذخیره حذف می‌شود.

## 4.5 merge اطلاعات محصول

ممکن است برای یک `asin` چند ردیف در `title_brand.csv` وجود داشته باشد. ابتدا کامل‌ترین ردیف بر اساس موجودبودن title و brand انتخاب می‌شود:

```python
product_df["_completeness"] = (
    product_df[["title", "brand"]].notna().sum(axis=1)
)
```

بعد `drop_duplicates("asin")` انجام می‌شود. گزینه زیر در merge اهمیت زیادی دارد:

```python
validate="many_to_one"
```

یعنی چند review می‌توانند به یک محصول متصل شوند، ولی برای هر `asin` در جدول محصول باید فقط یک ردیف وجود داشته باشد. اگر این قرارداد شکسته شود، pandas خطا می‌دهد و از تکثیر ناخواسته سطرها جلوگیری می‌کند.

title و brand در فایل خروجی حفظ می‌شوند، اما وارد `model_input` نمی‌شوند. دلیل آن نتیجه ablation داخلی پروژه است که افزودن آن‌ها Micro-F1 را کمی کاهش داد.

## 4.6 تبدیل verified

مقدار boolean به متن واضح تبدیل می‌شود:

```text
True  → yes
False → no
NaN   → unknown
```

مدل زبانی با متن کار می‌کند، بنابراین عبارت `Verified: yes` معنای روشن‌تری از مقدار خام boolean دارد.

## 4.7 پردازش vote

vote ممکن است به شکل رشته دارای comma باشد؛ مثلاً:

```text
"1,234"
```

ابتدا comma حذف و مقدار به عدد تبدیل می‌شود. مقادیر نامعتبر با `errors="coerce"` به NaN تبدیل می‌شوند.

به‌جای استفاده از عدد دقیق، vote به bucket تبدیل می‌شود:

```text
missing
0_to_4
5_to_9
10_to_49
50_plus
```

دلیل bucket کردن:

- توزیع vote بسیار راست‌چوله است.
- تفاوت 1 و 2 vote احتمالاً مهم نیست.
- مدل لازم نیست هزاران شکل عددی را یاد بگیرد.
- bucket معنای تقریبی محبوبیت review را منتقل می‌کند.

ردیف‌های فاقد vote حذف نمی‌شوند. فقط حدود 23٪ داده تمام مقادیر موردنظر را داشت و حذف missing vote امکان ساخت 50 هزار نمونه برای کلاس‌های 2 و 3 را از بین می‌برد. نبود vote با عبارت زیر نمایش داده می‌شود:

```text
Helpful votes: missing
```

این کار missingness را به یک ویژگی قابل‌آموزش تبدیل می‌کند.

## 4.8 محدودکردن summary

summary معمولاً کوتاه است، اما موارد غیرعادی ممکن است طولانی باشند. فقط 40 کلمه اول برای مدل نگه داشته می‌شود:

```python
def limit_words(value, max_words=40):
    return " ".join(str(value).split()[:max_words])
```

این محدودیت بر حسب کلمه است، نه token؛ چون tokenization واقعی داخل Notebook مدل انجام می‌شود. هدف این است که summary تمام بودجه 256 token را مصرف نکند.

## 4.9 ساخت model_input

ورودی نهایی:

```text
Verified: ... | Helpful votes: ... | Summary: ... | Review: ...
```

ترتیب فیلدها عمدی است. KerasHub رشته نهایی را از انتها truncate می‌کند. قرارگرفتن metadata کوتاه و summary در ابتدا باعث می‌شود همیشه حفظ شوند و فقط انتهای reviewهای بلند حذف شود.

چرا review در انتها قرار گرفت؟

- همه چهار نوع اطلاعات در ورودی حضور دارند.
- metadata فضای بسیار کمی مصرف می‌کند.
- summary به 40 کلمه محدود شده است.
- بخش اصلی بودجه همچنان در اختیار review است.

اگر در آینده tokenizer به‌صورت دستی استفاده شود، راه دقیق‌تر ساخت دو sequence و استفاده از truncation فقط روی review است. pipeline فعلی برای سادگی و سازگاری با KerasHub از یک رشته ساختاریافته استفاده می‌کند.

## 4.10 assertionها

assertion یک قرارداد اجرایی است. اگر فرض مهمی نقض شود، بهتر است pipeline متوقف شود تا اینکه فایل اشتباه بی‌صدا ساخته شود.

کنترل‌های اصلی:

- دقیقاً پنج کلاس وجود داشته باشد.
- هر کلاس 50 هزار سطر داشته باشد.
- تعداد کل 250 هزار باشد.
- exact duplicate باقی نمانده باشد.
- review خالی وجود نداشته باشد.
- `model_input` خالی یا NaN نباشد.
- پوشش metadata بیش از 99٪ باشد.
- validation هر کلاس دقیقاً 2 هزار باشد.
- train هر کلاس دقیقاً 48 هزار باشد.

## 4.11 پردازش test

test دقیقاً با همان cleaning و feature construction پردازش می‌شود. تفاوت‌های مهم:

- sampling انجام نمی‌شود.
- هیچ ردیفی حذف نمی‌شود.
- target اضافه نمی‌شود.
- ترتیب اصلی حفظ می‌شود.

پیش از merge یک ستون موقت ساخته می‌شود:

```python
test_df["_original_order"] = np.arange(test_rows)
```

بعد از merge، داده بر اساس آن مرتب و ستون حذف می‌شود. این کنترل حیاتی است، چون ترتیب predictionها باید با test خام یکسان باشد.

## 4.12 metadata.json

فایل metadata اطلاعات لازم برای reproducibility را ذخیره می‌کند:

- زمان ساخت
- مسیر source
- seed
- تعداد هر کلاس
- تعداد duplicateهای حذف‌شده
- تعداد متن‌های خالی حذف‌شده
- تعداد missing vote
- ترتیب فیلدهای مدل
- تعداد train، validation و test
- ستون‌های خروجی

بدون metadata ممکن است بعداً ندانیم یک مدل دقیقاً با کدام نسخه داده آموزش دیده است.

## 4.13 خروجی‌ها

```text
balanced_reviews.csv
train.csv
validation.csv
test.csv
metadata.json
```

---

# 5. نسخه Kaggle preprocessing

فایل: `notebooks/02_preprocessing_kaggle.ipynb`

منطق آن با نسخه محلی یکسان است، اما دو تفاوت دارد.

## 5.1 مستقل از src

تابع cleaning داخل خود Notebook تعریف شده است. دلیل آن این است که Dataset خام را بتوان مستقیم به Kaggle متصل کرد، بدون اینکه package محلی پروژه نصب شود.

برای حذف HTML از BeautifulSoup استفاده می‌شود. تابع فقط زمانی BeautifulSoup را صدا می‌زند که متن احتمالاً `<` و `>` داشته باشد؛ در نتیجه هزینه اضافی روی تمام سطرها تحمیل نمی‌شود.

## 5.2 کشف خودکار فایل‌ها

تابع `find_unique_file` داخل `/kaggle/input` دنبال فایل می‌گردد. فقط نام فایل کافی نیست؛ ستون‌های لازم نیز بررسی می‌شوند تا فایل اشتباه انتخاب نشود.

اگر بیش از یک candidate معتبر وجود داشته باشد، Notebook عمداً خطا می‌دهد. در این وضعیت باید مقدار زیر تنظیم شود:

```python
KAGGLE_DATASET_SLUG = "نام-dataset"
```

## 5.3 ساخت ZIP

علاوه بر CSVها، کل خروجی ZIP می‌شود:

```python
shutil.make_archive(...)
```

ZIP را می‌توان دانلود یا به‌عنوان Kaggle Dataset جدید publish کرد. Notebookهای مدل باید همین Dataset پردازش‌شده را دریافت کنند.

---

# 6. Notebook شماره 03: RoBERTa + LoRA با Keras

فایل: `notebooks/03_roberta_keras.ipynb`

هدف، fine-tune کردن `roberta-base` برای پنج کلاس است، ولی به‌جای تغییر همه وزن‌ها از LoRA استفاده می‌شود.

## 6.1 چرا RoBERTa؟

RoBERTa نسخه بهینه‌شده خانواده BERT است. برخلاف مدل‌های کلاسیک TF-IDF، می‌تواند معنای context، negation، ترتیب کلمات و شدت احساس را بهتر درک کند.

## 6.2 Keras Backend

قبل از import کردن Keras مقدار زیر تنظیم می‌شود:

```python
os.environ["KERAS_BACKEND"] = "tensorflow"
```

در Keras 3 backend می‌تواند TensorFlow، JAX یا PyTorch باشد. backend باید پیش از import Keras تعیین شود. این پروژه TensorFlow را انتخاب کرده چون محیط Kaggle و callbackهای فعلی بر آن بنا شده‌اند.

## 6.3 تنظیمات مدل

```python
NUM_CLASSES = 5
MAX_LENGTH = 256
EPOCHS = 5
PER_REPLICA_BATCH_SIZE = 8
LORA_RANK = 16
PEAK_LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
```

### MAX_LENGTH=256

طول 128 سریع‌تر بود، اما بخش بیشتری از review را حذف می‌کرد. طول 256 توازن بهتری میان اطلاعات متن، حافظه GPU و زمان آموزش ایجاد می‌کند.

### Batch size=8

با sequence طولانی‌تر، activation بیشتری در GPU ذخیره می‌شود. batch 8 برای P100 شانزده گیگابایتی مقدار محافظه‌کارانه‌ای است. اگر OOM رخ دهد، اولین تغییر باید batch 4 باشد.

### EPOCHS=5

این مقدار سقف آموزش است، نه الزام اجرای هر پنج epoch. EarlyStopping می‌تواند زودتر آموزش را متوقف و بهترین وزن‌ها را بازگرداند.

## 6.4 تک GPU به‌جای MirroredStrategy

در ترکیب TensorFlow 2.20، KerasHub 0.26 و LoRA، استفاده از دو T4 با MirroredStrategy باعث خطای reduction روی متغیر boolean شد:

```text
Value for attr 'T' of bool ... AddN
```

به همین دلیل:

```python
USE_MULTI_GPU = False
strategy = tf.distribute.get_strategy()
```

مدل همچنان روی GPU اجرا می‌شود، اما cross-replica reduction انجام نمی‌شود. روی P100 که یک GPU است این انتخاب طبیعی است.

## 6.5 Mixed precision

```python
keras.mixed_precision.set_global_policy("mixed_float16")
```

بخش زیادی از محاسبات با float16 انجام می‌شود. این کار حافظه و زمان را کاهش می‌دهد. Keras بخش‌های حساس مانند loss scaling را مدیریت می‌کند.

P100 Tensor Core نسل T4 را ندارد؛ بنابراین سود سرعت mixed precision ممکن است کمتر باشد، ولی کاهش حافظه همچنان مفید است.

## 6.6 tf.data pipeline

train dataset:

```text
from_tensor_slices
→ shuffle
→ batch
→ prefetch
```

`shuffle` مانع دیدن کلاس‌ها در یک ترتیب ثابت می‌شود. `reshuffle_each_iteration=True` در هر epoch ترتیب تازه‌ای می‌سازد.

`drop_remainder=True` در train باعث می‌شود تمام batchها شکل یکسان داشته باشند. validation هیچ ردیفی را drop نمی‌کند.

`prefetch(AUTOTUNE)` آماده‌سازی batch بعدی را با محاسبه batch فعلی هم‌پوشان می‌کند.

## 6.7 label mapping

امتیازهای اصلی 1 تا 5 هستند، ولی loss کلاس‌ها را از صفر انتظار دارد:

```text
1..5 → 0..4
```

برای گزارش نهایی دوباره یک واحد اضافه می‌شود.

## 6.8 LoRA چیست؟

در full fine-tuning همه ماتریس وزن تغییر می‌کنند. LoRA وزن اصلی را freeze می‌کند و تغییر وزن را با دو ماتریس کم‌رتبه یاد می‌گیرد:

```text
W جدید = W ثابت + A × B
```

اگر rank کوچک باشد، تعداد پارامترهای `A` و `B` بسیار کمتر از `W` است.

مزایا:

- optimizer state کوچک‌تر
- فایل adapter کوچک
- حافظه کمتر
- امکان نگهداری چند adapter برای یک backbone

در Notebook:

```python
model.backbone.enable_lora(rank=16)
```

rank 16 نسبت به rankهای بسیار کوچک ظرفیت بیشتری برای یادگیری تفاوت ظریف امتیازهای 2، 3 و 4 دارد.

## 6.9 بارگذاری preset در Kaggle

Kaggle در اجرای non-interactive نمی‌تواند مدل جدید را خودکار attach کند. بنابراین مدل باید از Add Input → Models به Notebook متصل شود.

تابع resolver ابتدا مسیر محلی `/kaggle/input` را می‌گردد. اگر مدل محلی پیدا نشود و Internet نیز خاموش باشد، خطای توضیحی تولید می‌کند.

این طراحی از خطای مبهم KaggleHub جلوگیری می‌کند.

## 6.10 AdamW

optimizer:

```python
keras.optimizers.AdamW(...)
```

Adam برای هر پارامتر میانگین و واریانس gradient را نگه می‌دارد و learning rate مؤثر را تطبیق می‌دهد. AdamW weight decay را به‌شکل صحیح از update گرادیان جدا می‌کند.

پارامترهای bias و normalization از weight decay مستثنا می‌شوند:

```python
optimizer.exclude_from_weight_decay(
    var_names=["bias", "beta", "gamma"]
)
```

اعمال decay روی این پارامترها معمولاً مفید نیست.

`global_clipnorm=1.0` gradientهای بسیار بزرگ را محدود و آموزش را پایدارتر می‌کند.

## 6.11 Warmup و Cosine Decay

learning rate از صفر شروع و طی 10٪ کل stepها به مقدار peak می‌رسد. سپس با منحنی cosine کاهش می‌یابد.

Warmup از تغییر شدید وزن‌ها در ابتدای آموزش جلوگیری می‌کند. decay نیز در مراحل پایانی updateها را ظریف‌تر می‌کند.

نکته تفسیر log: در RoBERTa با 30 هزار step در epoch و 5 epoch:

```text
Total steps  = 150,000
Warmup steps = 15,000
```

بنابراین metricهای نیمه اول epoch اول هنوز مربوط به learning rate پایین هستند و نباید نتیجه نهایی تلقی شوند.

## 6.12 Compile و loss

خروجی مدل logits خام است:

```python
activation=None
```

پس loss با گزینه زیر ساخته می‌شود:

```python
SparseCategoricalCrossentropy(from_logits=True)
```

استفاده هم‌زمان از softmax در مدل و `from_logits=True` اشتباه است. در این پروژه softmax داخل loss به‌صورت عددی پایدار اعمال می‌شود.

Metric زمان train با نام `micro_f1` در واقع SparseCategoricalAccuracy است. این نام‌گذاری به دلیل برابر بودن Accuracy و Micro-F1 در این مسئله تک‌برچسبی انتخاب شده است. پس از predict، F1 واقعی با scikit-learn نیز محاسبه می‌شود.

## 6.13 callbackها

### BackupAndRestore

وضعیت train شامل وزن‌ها، optimizer و پیشرفت fit را هر 500 batch ذخیره می‌کند. برای resume دقیق، این پوشه از `latest.weights.h5` مهم‌تر است.

### latest.weights.h5

هر 500 batch فقط وزن فعلی را ذخیره می‌کند. اگر training backup خراب شود، این فایل fallback است. با آن می‌توان وزن‌ها را بازیابی کرد، ولی optimizer و موقعیت schedule از دست می‌روند.

### best.weights.h5

پس از هر epoch، بهترین وزن بر اساس `val_micro_f1` ذخیره می‌شود. مدل نهایی لزوماً آخرین epoch نیست.

### EarlyStopping

اگر metric برای دو epoch بهتر نشود، آموزش متوقف و بهترین وزن بازیابی می‌شود.

### CSVLogger و TensorBoard

CSVLogger تاریخچه عددی epochها را ذخیره می‌کند. TensorBoard برای مشاهده نمودار loss و metric است.

### TerminateOnNaN

اگر loss به NaN تبدیل شود، آموزش فوراً متوقف می‌شود تا GPU بی‌دلیل ساعت‌ها کار نکند.

## 6.14 ارزیابی

پس از fit، best checkpoint بارگذاری می‌شود. logits با `argmax` به کلاس تبدیل می‌شوند.

موارد زیر تولید می‌شود:

- Accuracy
- Micro-F1
- Macro-F1
- Weighted-F1
- precision/recall/F1 هر کلاس
- confusion matrix
- CSV پیش‌بینی validation

Confusion matrix نشان می‌دهد مدل کدام کلاس‌ها را با هم اشتباه می‌گیرد؛ در این پروژه خطاهای مجاور مانند 2↔3 و 4↔5 اهمیت ویژه دارند.

## 6.15 ذخیره artifact

سه خروجی مدل ذخیره می‌شود:

```text
roberta_lora_adapters.lora.h5
roberta_lora_rating_classifier.keras
roberta_lora_rating_classifier_preset/
```

KerasHub الزام می‌کند نام adapter دقیقاً به `.lora.h5` ختم شود.

---

# 7. Notebook شماره 04: DeBERTa-v3 + LoRA

فایل: `notebooks/04_deberta_v3_keras.ipynb`

ساختار آزمایش عمداً تقریباً با RoBERTa یکسان است تا مقایسه controlled باشد.

## 7.1 چرا DeBERTa-v3؟

DeBERTa از disentangled attention استفاده می‌کند؛ یعنی اطلاعات محتوای token و موقعیت آن را جداگانه مدل می‌کند. نسخه V3 از روش pretraining شبیه ELECTRA نیز بهره می‌برد.

هدف این آزمایش این است که ببینیم معماری قوی‌تر NLU می‌تواند تفاوت شدت احساس میان امتیازهای مجاور را بهتر یاد بگیرد یا نه.

## 7.2 تنظیمات متفاوت

```python
MODEL_PRESET = "deberta_v3_base_en"
EPOCHS = 4
PEAK_LEARNING_RATE = 4.5e-5
```

learning rate بر اساس محدوده پیشنهادی آزمایش‌های رسمی DeBERTa انتخاب شده است. سایر اجزای مهم مانند split، max length، rank و metricها تا حد ممکن ثابت نگه داشته شده‌اند.

## 7.3 هزینه محاسباتی

DeBERTa-v3-base از RoBERTa کندتر است. در log مشاهده‌شده:

```text
RoBERTa:  حدود 240 ms/step
DeBERTa: حدود 634 ms/step
```

پس DeBERTa تقریباً 2.6 برابر کندتر بود. با 30 هزار step در epoch، یک epoch می‌تواند بیش از پنج ساعت طول بکشد.

## 7.4 تفسیر metric ابتدای DeBERTa

برای پنج کلاس، baseline تصادفی:

```text
loss ≈ ln(5) ≈ 1.609
accuracy ≈ 0.20
```

در stepهای ابتدایی DeBERTa مقدار `loss≈1.626` و `micro_f1≈0.202` دیده شد. این الزاماً خرابی نیست؛ چون warmup آن 12 هزار step است و در step 1000 learning rate هنوز بسیار کوچک است.

قضاوت باید پس از عبور از بخش معناداری از warmup و ترجیحاً پس از validation پایان epoch انجام شود.

## 7.5 Resume در محیط دیگر

برای ادامه دقیق آموزش باید این پوشه حفظ شود:

```text
deberta_v3_keras_lora/training_backup/
```

در محیط جدید باید موارد زیر یکسان باشند:

- نسخه TensorFlow/Keras/KerasHub
- معماری و preset
- optimizer و schedule
- batch size و تعداد stepها
- Dataset و pipeline
- مسیر `BACKUP_DIR`

فایل `latest.weights.h5` فقط وزن‌ها را بازمی‌گرداند و resume دقیق optimizer نیست.

پیش از رسیدن به محدودیت Kaggle بهتر است بعد از پیام موفق checkpoint، training دستی متوقف و کل output ZIP شود.

---

# 8. روش صحیح مقایسه RoBERTa و DeBERTa

برای مقایسه معتبر باید ثابت باشند:

```text
train.csv
validation.csv
model_input
label mapping
MAX_LENGTH
seed
LoRA rank
معیارهای ارزیابی
```

مواردی که باید ثبت شوند:

| مورد | دلیل |
|---|---|
| Micro-F1 | معیار اصلی پروژه |
| Macro-F1 | کیفیت کلاس‌های دشوار |
| Weighted-F1 | عملکرد وزن‌دار بر حسب support |
| F1 هر کلاس | تشخیص ضعف کلاس‌های 2 و 3 |
| Runtime | هزینه عملی مدل |
| Trainable parameters | میزان بهره‌وری LoRA |
| Adapter size | هزینه ذخیره و انتقال |
| Confusion matrix | نوع خطاها |

مدل قوی‌تر فقط مدلی نیست که 0.001 امتیاز بهتر باشد؛ اگر اختلاف بسیار کوچک ولی زمان آموزش چند برابر باشد، RoBERTa ممکن است انتخاب عملی بهتری باشد.

---

# 9. خطاهای مهمی که تاکنون حل شدند

## خطای attach مدل Kaggle

```text
New Models cannot be attached in non-interactive sessions
```

راه‌حل: مدل باید پیش از Save & Run All از Add Input → Models متصل شود.

## خطای boolean AddN

```text
Value for attr 'T' of bool ... AddN
```

علت: ناسازگاری cross-replica reduction در MirroredStrategy با stack فعلی KerasHub/LoRA.

راه‌حل:

```python
USE_MULTI_GPU = False
```

## خطای پسوند LoRA

```text
The filename must end in `.lora.h5`
```

راه‌حل: نام adapter باید دقیقاً چنین پسوندی داشته باشد.

## خطای review خالی

```python
assert balanced_df[TEXT_COLUMN].str.strip().ne("").all()
```

علت: sampling پیش از حذف reviewهای خالی انجام شده بود.

راه‌حل: eligibility قبل از sampling و پس از cleaning بررسی شد.

## هشدار use_unbounded_threadpool

این پیام از تفاوت binary تولیدکننده graph و تعریف op در TensorFlow می‌آید. اگر training ادامه پیدا کند، هشدار است و خطای اصلی محسوب نمی‌شود.

---

# 10. مراحل بعدی پروژه

پس از پایان مدل‌ها:

1. بهترین checkpoint هر مدل بارگذاری شود.
2. هر دو روی validation مشترک ارزیابی شوند.
3. جدول مقایسه ساخته شود.
4. مدل نهایی فقط بر اساس validation انتخاب شود.
5. `test.csv` پردازش‌شده بدون تغییر ترتیب وارد مدل شود.
6. خروجی 0 تا 4 به 1 تا 5 تبدیل شود.
7. submission ساخته شود:

```python
submission = pd.DataFrame({
    "predicted": test_predictions + 1,
})

submission.to_csv("q2_submission.csv", index=False)
```

کنترل‌های نهایی:

```python
assert len(submission) == len(test_df)
assert submission.columns.tolist() == ["predicted"]
assert submission["predicted"].between(1, 5).all()
assert not submission["predicted"].isna().any()
```

test نباید برای انتخاب epoch، learning rate یا مدل استفاده شود. استفاده از test برای model selection نوعی leakage است، حتی اگر labelهای آن در دسترس قرار گیرند.

---

# 11. چک‌لیست اجرای دوباره پروژه

## ساخت Dataset

- فایل‌های خام train، test و title_brand به Kaggle متصل باشند.
- `02_preprocessing_kaggle.ipynb` اجرا شود.
- assertionها همگی پاس شوند.
- ZIP یا Dataset خروجی publish شود.
- خروجی شامل train، validation، test و metadata باشد.

## آموزش مدل

- Dataset پردازش‌شده متصل باشد.
- preset مدل از Kaggle Models متصل باشد.
- P100 فعال باشد.
- `USE_MULTI_GPU=False` باشد.
- log تعداد train و validation درست باشد.
- train برابر 240 هزار و validation برابر 10 هزار باشد.
- checkpointها واقعاً در `/kaggle/working` ساخته شوند.

## پیش از محدودیت Session

- منتظر پایان موفق checkpoint بمانید.
- training را دستی متوقف کنید.
- `training_backup` و `checkpoints` را ZIP کنید.
- فایل ZIP را دانلود یا به output یک نسخه Kaggle تبدیل کنید.

## ارزیابی نهایی

- best checkpoint، نه صرفاً latest checkpoint، استفاده شود.
- Micro-F1 و Macro-F1 هر دو گزارش شوند.
- classification report و confusion matrix ذخیره شوند.
- نتایج فقط روی validation مقایسه شوند.

این سند باید همراه با تغییر Notebookها به‌روزرسانی شود تا تصمیم‌های پروژه و دلیل آن‌ها برای اجرای آینده قابل‌ردیابی باقی بماند.
