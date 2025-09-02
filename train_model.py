# train_model.py
import io, zipfile
from pathlib import Path
import requests, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_PATH = Path("spam_pipeline.joblib")

def ensure_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_FILE.exists():
        print("📦 Dataset ya presente.")
        return
    print("📥 Descargando dataset...")
    resp = requests.get(URL, timeout=30); resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(DATA_DIR)
    print("✅ Dataset extraído en:", DATA_DIR.resolve())

def load_df():
    df = pd.read_csv(RAW_FILE, sep="\t", names=["label", "message"])
    df = df[df["label"].isin(["ham","spam"])].copy()
    df["message"] = df["message"].astype(str)
    print(df.head()); print("\nDistribución de clases:\n", df["label"].value_counts())
    return df

def main():
    ensure_dataset()
    df = load_df()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=0.3, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1,2), min_df=2
        )),
        ("nb", MultinomialNB(alpha=0.5))
    ])

    print("\nEntrenando...")
    pipe.fit(X_train, y_train)

    print("\nEvaluando...")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(pipe, MODEL_PATH)
    print(f"\n💾 Modelo guardado en: {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
