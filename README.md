🛒 E-Commerce Purchase Intent Prediction

Predicting whether a website visitor will make a purchase using machine learning — built to solve a real business problem in e-commerce conversion optimization.

## 📌 Problem Statement

Out of every 100 visitors on an e-commerce website, only about 15 actually make a purchase. The rest leave without buying. Can we predict — using session behavior data — which visitors are likely to convert?

This project builds a binary classification model that answers exactly that question, helping businesses identify high-intent visitors and take targeted action to improve conversion rates.

---

## 📂 Dataset

**Source:** [Online Shoppers Purchasing Intention Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

| Attribute | Details |
|-----------|---------|
| Total Sessions | 12,330 |
| Features | 17 (behavioral + technical) |
| Target Variable | `Revenue` — True (Purchase) / False (No Purchase) |
| Class Distribution | 84.5% No Purchase / 15.5% Purchase |
| Missing Values | None |

---

## ⚙️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Class Imbalance | Imbalanced-learn (SMOTE) |
| Environment | Jupyter Notebook |

---

## 🔍 Project Workflow

```
Data Loading → EDA → Preprocessing → SMOTE → Model Building → Tuning → Evaluation → Insights
```

### 1. Exploratory Data Analysis
- Identified **class imbalance** — only 15.5% of sessions result in purchase
- Found **November** has the highest purchase volume (festive season effect)
- **Returning visitors** convert at a higher rate than new visitors
- Sessions ending in purchase have significantly **lower bounce and exit rates**
- `PageValues` showed the strongest separation between buyers and non-buyers

### 2. Preprocessing
- Label Encoding for categorical columns: `Month`, `VisitorType`, `Weekend`, `Revenue`
- Train-test split: **80/20** with `stratify=Revenue` and `random_state=42`
- `StandardScaler` applied to training data only for Logistic Regression
- Tree-based models used unscaled data (scale-invariant)

### 3. Handling Class Imbalance — SMOTE
- Applied SMOTE **only on training data** (never on test set)
- Balanced training set: 8,332 vs 8,332 sessions
- Test set kept at real-world distribution for honest evaluation

### 4. Models Built

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline — requires feature scaling |
| Decision Tree | Default (overfitting demo) → tuned via GridSearchCV |
| Random Forest | Ensemble of 100 trees — best performer |

### 5. Overfitting Analysis
Trained Decision Tree across depths 1–20 and plotted train vs test accuracy.
Demonstrated that an unconstrained tree hits **100% training accuracy** while test accuracy drops — classic overfitting.

### 6. Hyperparameter Tuning — GridSearchCV
- Searched 108 combinations across `criterion`, `max_depth`, `min_samples_split`
- 5-fold cross-validation with **F1 as scoring metric** (chosen over accuracy due to class imbalance)
- Best params: `criterion=gini`, `max_depth=10`, `min_samples_split=40`
- Best CV F1 Score: **0.90**

### 7. Feature Importance & Selection
- Used Random Forest feature importances to rank all 17 features
- Retrained model on **top 10 features** — matched full model performance
- Confirms 7 features contributed minimal signal

**Top 5 Features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | PageValues | 0.4883 |
| 2 | ExitRates | 0.0872 |
| 3 | ProductRelated_Duration | 0.0676 |
| 4 | Administrative_Duration | 0.0613 |
| 5 | ProductRelated | 0.0458 |

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.86 | 0.55 | 0.70 | 0.62 | 0.86 |
| Decision Tree (GridSearch) | 0.86 | 0.54 | 0.71 | 0.61 | 0.87 |
| **Random Forest** | **0.88** | **0.60** | **0.72** | **0.66** | **0.91** |

> ✅ **Random Forest** is the best model with AUC-ROC of **0.91**

All metrics calculated on the held-out test set (2,466 sessions) using the actual class distribution — not SMOTE-balanced — for a fair, real-world evaluation.

---

## 💡 Business Insights

- **PageValues is the #1 predictor** (48.8% importance) — optimizing high-value product pages directly impacts conversion
- **Lower exit rates = more purchases** — reducing exit rates through better UI/UX is a high-ROI improvement
- **November drives the most purchases** — marketing spend should be intensified in Oct–Nov
- **Returning visitors convert more** — loyalty and re-engagement programs can leverage this pattern
- **Feature selection** confirmed 7 features were noise — a leaner 10-feature model matched full performance

> 💬 *Business Recommendation: Deploy the Random Forest model to score live sessions in real-time. Flag high-probability buyers for targeted interventions — personalized discounts, cart reminders, or live chat — to maximize conversion without wasting spend on low-intent visitors.*

---

## 📁 Repository Structure

```
ecommerce-purchase-intent-prediction/
│
├── ecommerce-purchase-intent-prediction.ipynb ← Main notebook (full analysis)
├── online_shoppers_intention.csv      ← Dataset
├── README.md                          ← You are here
└── Online_Retail_Conversion_Prediction.docx  ← Detailed project report
``
