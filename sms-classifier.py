import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "SMSSpamCollection"

def ensure_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_FILE.exists():
        print("📦 Dataset ya presente en:", RAW_FILE.resolve())
        return
    print("📥 Descargando dataset...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(DATA_DIR)
    print("✅ Dataset descargado y extraído en:", DATA_DIR.resolve())

def load_df() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILE, sep="\t", names=["label", "message"])
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["message"] = df["message"].astype(str)
    print(df.head())
    print("\nDistribución de clases:")
    print(df["label"].value_counts())
    return df

def train_vectorize(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=0.3, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",   # buen baseline para inglés
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=2                # ignora términos muy raros
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\nResultados del modelo (Naive Bayes con dataset real):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # helper para predecir mensajes nuevos
    def predict_message(msg: str) -> str:
        vec = vectorizer.transform([msg])
        return model.predict(vec)[0]

    # demo rápida
    print("\nPredicciones con mensajes nuevos:")
    for s in [
        "Win a free iPhone now!",
        "Hey, wanna go to the movies later?",
        "Please update your account information to avoid suspension."
    ]:
        print(f"{s} -> {predict_message(s)}")

def main():
    ensure_dataset()
    df = load_df()
    train_vectorize(df)

if __name__ == "__main__":
    main()

