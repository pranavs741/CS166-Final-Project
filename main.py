"""
CS 166 Phishing and Smishing Detector
Group Project
"""

import os
import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

from wordcloud import WordCloud


DATA_PATH = "data.csv"
OUT_DIR = "outputs"
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")


# load data
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    print(df['label'].value_counts())
    print(df['source'].value_counts())
    return df


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = df.groupby(['source', 'label']).size().unstack()
    counts.plot(kind='bar', ax=ax, color=['#3CB371', '#DC3545'])
    ax.set_title("Sample Counts by Source and Label", fontsize=13)
    ax.set_xlabel("Message Source")
    ax.set_ylabel("Count")
    ax.set_xticklabels(['Email', 'SMS'], rotation=0)
    ax.legend(title="Label")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
    plt.close()
    print("  Saved class_balance.png")


def plot_length_distribution(df):
    df['length'] = df['content'].str.len()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in [('legitimate', '#3CB371'), ('phishing', '#DC3545')]:
        subset = df[df['label'] == label]['length']
        ax.hist(subset, bins=20, alpha=0.6, label=label, color=color)
    ax.set_title("Message Length Distribution", fontsize=13)
    ax.set_xlabel("Length (characters)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/length_distribution.png", dpi=150)
    plt.close()
    print("  Saved length_distribution.png")


# word clouds
def make_wordcloud(text, title, filename, color):
    wc = WordCloud(width=900, height=500,
                   background_color="white",
                   colormap=color,
                   max_words=80,
                   stopwords=None,
                   collocations=False).generate(text)
    plt.figure(figsize=(9, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def generate_wordclouds(df):
    phish_text = " ".join(df[df['label'] == 'phishing']['content'].tolist())
    legit_text = " ".join(df[df['label'] == 'legitimate']['content'].tolist())

    make_wordcloud(phish_text, "Phishing & Smishing Word Cloud",
                   "wordcloud_phishing.png", "Reds")
    make_wordcloud(legit_text, "Legitimate Word Cloud",
                   "wordcloud_legitimate.png", "Greens")


# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' urlplaceholder ', text)
    text = re.sub(r'\d+', ' numplaceholder ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_handcrafted_features(text):
    raw = text
    feats = {
        'num_urls': len(re.findall(r'http\S+|www\.\S+', raw)),
        'num_digits': sum(c.isdigit() for c in raw),
        'num_exclaim': raw.count('!'),
        'num_dollar': raw.count('$') + raw.count('£'),
        'num_caps_words': sum(1 for w in raw.split() if w.isupper() and len(w) > 1),
        'length': len(raw),
    }
    return feats


# train models
def train_and_evaluate(df):
    df['cleaned'] = df['content'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=2000,
                                 ngram_range=(1, 2),
                                 stop_words='english',
                                 min_df=1)
    X = vectorizer.fit_transform(df['cleaned'])
    y = (df['label'] == 'phishing').astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set: {X_train.shape[0]}  |  Test set: {X_test.shape[0]}")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cm': confusion_matrix(y_test, y_pred),
            'model': model,
        }
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred,
                                    target_names=['legitimate', 'phishing']))

    return results, vectorizer, X_test, y_test


# plots
def plot_model_comparison(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())
    data = np.array([[results[m][k] for k in metrics] for m in model_names])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for i, name in enumerate(model_names):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[i], width, label=name, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha='center', fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print("  Saved model_comparison.png")


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legit', 'Phish'],
                    yticklabels=['Legit', 'Phish'],
                    cbar=False, ax=ax)
        ax.set_title(f"{name}\nAcc = {res['accuracy']:.2f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()
    print("  Saved confusion_matrices.png")


def plot_top_features(results, vectorizer, top_n=15):
    lr = results['Logistic Regression']['model']
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr.coef_[0]

    top_phish = np.argsort(coefs)[-top_n:]
    top_legit = np.argsort(coefs)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(feature_names[top_phish], coefs[top_phish], color='#DC3545')
    axes[0].set_title("Top Phishing Indicators")
    axes[0].set_xlabel("LR Coefficient")

    axes[1].barh(feature_names[top_legit], coefs[top_legit], color='#3CB371')
    axes[1].set_title("Top Legitimate Indicators")
    axes[1].set_xlabel("LR Coefficient")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top_features.png", dpi=150)
    plt.close()
    print("  Saved top_features.png")


# predict
def predict_message(text, results, vectorizer):
    cleaned = clean_text(text)
    X_new = vectorizer.transform([cleaned])
    lr = results['Logistic Regression']['model']
    pred = lr.predict(X_new)[0]
    proba = lr.predict_proba(X_new)[0]
    label = 'PHISHING' if pred == 1 else 'LEGITIMATE'
    confidence = proba[pred]
    return label, confidence


def demo_predictions(results, vectorizer):
    test_messages = [
        "Hey, are you free for lunch tomorrow at 12?",
        "URGENT! You have WON $1000! Call 09050000123 NOW to claim!",
        "Your Amazon package will be delivered tomorrow between 2-4 PM.",
        "Dear customer, your account has been suspended. Click here to verify: http://amaz0n-secure-login.tk/auth",
    ]
    print("\n" + "=" * 60)
    print("LIVE DEMO PREDICTIONS")
    print("=" * 60)
    for msg in test_messages:
        label, conf = predict_message(msg, results, vectorizer)
        print(f"\n[{label}]  (confidence: {conf:.0%})")
        print(f"  {msg}")


def main():
    print("=" * 60)
    print("CS 166 Phishing and Smishing Detector")
    print("=" * 60)

    df = load_data(DATA_PATH)

    print("\n[Step 1] Generating EDA plots...")
    plot_class_balance(df)
    plot_length_distribution(df)

    print("\n[Step 2] Generating word clouds...")
    generate_wordclouds(df)

    print("\n[Step 3] Training models...")
    results, vectorizer, X_test, y_test = train_and_evaluate(df)

    print("\n[Step 4] Generating result plots...")
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_top_features(results, vectorizer)

    print("\n[Step 5] Demo predictions:")
    demo_predictions(results, vectorizer)

    print("\nDone! All outputs saved to:", OUT_DIR + "/")


if __name__ == "__main__":
    main()
