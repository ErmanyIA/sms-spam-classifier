# predict.py
import sys, joblib
from pathlib import Path

MODEL_PATH = Path("spam_pipeline.joblib")

def main():
    if len(sys.argv) < 2:
        print('Uso: python predict.py "tu mensaje aquí"')
        sys.exit(1)

    if not MODEL_PATH.exists():
        print("No encuentro el modelo. Entrena primero: python train_model.py")
        sys.exit(1)

    msg = " ".join(sys.argv[1:])
    pipe = joblib.load(MODEL_PATH)

    pred = pipe.predict([msg])[0]
    proba = pipe.predict_proba([msg])[0].max()

    print(f"Predicción: {pred}  (confianza={proba:.2f})")

if __name__ == "__main__":
    main()
