import re
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try: nltk.data.find("tokenizers/punkt")
except: nltk.download("punkt")
try: nltk.data.find("corpora/stopwords")
except: nltk.download("stopwords")
try: nltk.data.find("corpora/wordnet")
except: nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

def clean_text(t):
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tok = re.split(r"\s+", t)
    tok = [w for w in tok if w and w not in STOPWORDS and len(w) > 1]
    tok = [LEM.lemmatize(w) for w in tok]
    return " ".join(tok)

def load_dataset(path="news.csv"):
    df = pd.read_csv(path)
    if "text" in df.columns: df["content"] = df["text"].astype(str)
    elif "article" in df.columns: df["content"] = df["article"].astype(str)
    elif "title" in df.columns: df["content"] = df["title"].astype(str)
    else:
        c = df.columns[0]
        df["content"] = df[c].astype(str)

    if "label" in df.columns: lbl = "label"
    elif "target" in df.columns: lbl = "target"
    elif "class" in df.columns: lbl = "class"
    else: lbl = df.columns[-1]

    def map_label(x):
        if isinstance(x, str):
            s = x.lower().strip()
            if s in ("fake","0","false"): return 0
            if s in ("real","1","true"): return 1
        try: return 1 if int(x) == 1 else 0
        except: return 0

    df["label"] = df[lbl].apply(map_label)
    return df[["content","label"]]

def main(dataset="news.csv", model_dir="models"):
    df = load_dataset(dataset)
    df["clean"] = df["content"].map(clean_text)
    X = df["clean"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if not os.path.exists(model_dir): os.makedirs(model_dir)

    vect = TfidfVectorizer(max_features=25000, ngram_range=(1,2))
    X_train_tf = vect.fit_transform(X_train)
    X_test_tf = vect.transform(X_test)

    nb = MultinomialNB().fit(X_train_tf, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test_tf))

    lr = LogisticRegression(max_iter=1000, solver="liblinear").fit(X_train_tf, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_tf))

    best = lr if lr_acc >= nb_acc else nb

    joblib.dump(vect, os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(best, os.path.join(model_dir, "best_model.joblib"))

    preds = best.predict(X_test_tf)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["FAKE","REAL"]))
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    d = "news.csv"
    if len(sys.argv) > 1: d = sys.argv[1]
    main(dataset=d)
