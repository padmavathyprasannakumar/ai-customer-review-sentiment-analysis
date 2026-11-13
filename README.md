# AI-Based Customer Review & Sentiment Analysis

Analyses customer reviews, classifies each one as **Positive, Neutral or Negative**, and visualises customer satisfaction trends for an India-based superstore.

**Tools:** Python · Pandas · NLP (TF-IDF) · Scikit-learn · Matplotlib · Seaborn · WordCloud · Streamlit

🔗 **Live app:** _add your Streamlit link here after deployment_

---

## Business problem

Customers leave thousands of reviews. Reading them all by hand is slow, so the business cannot quickly see:

1. Whether customers are happy or unhappy overall, and whether this is improving.
2. Which products, months and regions get the most complaints.
3. **Why** customers are unhappy.

This project uses NLP and machine learning to answer these questions automatically.

## Dataset

- **7,025 customer reviews**, January 2022 – December 2025
- Linked to the orders, customers, products and cities of the Project 1 sales data
- Main columns: Review Date, Rating (1–5), Review Title, **Review Text**, Category, Sub-Category, City, Region
- After removing 35 empty reviews, 25 duplicates and 17 reviews with no meaningful words: **6,948 reviews**

**Sentiment labels from star ratings:** 4–5 ⭐ → Positive · 3 ⭐ → Neutral · 1–2 ⭐ → Negative
Result: 57% Positive, 24% Neutral, 19% Negative.

## Project steps

| # | Step | Notebook / file |
|---|------|-----------------|
| 1 | Project setup and dataset | `data/` |
| 2–3 | Review exploration and sentiment labels | `notebooks/01_review_exploration.ipynb` |
| 4–5 | Text cleaning and TF-IDF | `notebooks/02_text_cleaning_and_tfidf.ipynb` |
| 6 | Model training and evaluation | `notebooks/03_sentiment_model.ipynb` |
| 7 | Satisfaction trends and complaint analysis | `notebooks/04_sentiment_trends.ipynb` |
| 8 | Streamlit web app | `app/app.py` |

## Text cleaning (NLP)

- Lowercase, remove punctuation and numbers
- Remove common stopwords (*the, is, for…*) using scikit-learn's English list
- **Keep negation and intensity words** such as *not, no, never, very, too*, so "not good" is not reduced to "good"
- Average review length fell from 9.9 to 6.3 words

## Turning text into numbers: TF-IDF

Each review becomes a vector of TF-IDF scores: a word scores high when it is important in that review but rare across all reviews. Single words and **word pairs** (bigrams such as *not good*, *not happy*) are used. TF-IDF is fitted on the training data only, inside a scikit-learn `Pipeline`, to avoid data leakage.

## Model results

80/20 stratified train/test split (5,558 training, 1,390 test reviews). `class_weight="balanced"` was used for the models that support it, because the classes are imbalanced.

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Naive Bayes** | **89.8%** | **0.867** |
| Logistic Regression | 89.4% | 0.865 |
| Linear SVM | 88.9% | 0.859 |

**Best model: Naive Bayes**

| Sentiment | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| Negative | 0.87 | 0.95 | 0.91 |
| Neutral | 0.90 | 0.64 | 0.75 |
| Positive | 0.91 | 0.99 | 0.94 |

- The model never confused Positive with Negative.
- **Neutral is the hardest class**, because neutral reviews mix positive and negative words ("Good product but delivery was late").

## Key insights

- **Satisfaction is improving:** satisfaction score (% positive − % negative) rose from 32.4 (2022) to 42.2 (2025).
- **Festive season complaints:** negative reviews peak in October–December (about 22–23%), during the Diwali sales rush.
- **East region is least satisfied:** Ranchi, Bhubaneswar, Guwahati, Patna and Kolkata have the highest share of negative reviews.
- **Furniture, especially Tables,** gets the most complaints (Tables also lost money in Project 1).
- **Top complaint reasons** (share of negative reviews): delivery delay 37.7%, damaged or defective 27.6%, customer service 20.0%, poor quality 19.6%, price/value 19.1%, refund/return 16.1%, packaging 11.1%.

## Recommendations

1. **Fix delivery first:** add delivery capacity before Diwali, when complaints rise.
2. **Review courier and warehouse operations in the East region.**
3. **Check quality and packaging for Furniture, especially Tables.**
4. **Improve packaging for fragile items** to reduce damage complaints.

## Streamlit web app

- **Analyse a Review:** type a review and get its sentiment with a confidence score.
- **Analyse a File:** upload a CSV of reviews, get the sentiment for every row and download the results.
- **Satisfaction Dashboard:** category and region filters, KPIs, sentiment trend, satisfaction score by year, complaint reasons and word clouds.

## Project structure

```
sentiment_analysis/
├── app/
│   └── app.py                     # Streamlit web app
├── data/                          # raw, labelled and cleaned review data
├── models/
│   └── sentiment_model.pkl        # TF-IDF + Naive Bayes pipeline
├── notebooks/                     # 01 to 04, in order
├── requirements.txt
└── README.md
```

## How to run

```
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Then open http://localhost:8501.

## Limitations and future work

- Star ratings are used as labels, and some reviewers' stars don't match their words.
- The model can miss sarcasm (for example "Great, it broke on day one").
- Future ideas: explain each prediction with its most influential words, compare sentiment across Project 1 customer segments, and try pretrained language models.

---

**Author:** Padmavathy G · AI Data Science Intern, Vetri IT Systems Pvt. Ltd.