# SMS Spam Classifier (TF‑IDF + Naive Bayes)

Clasifica mensajes SMS como **spam** o **ham** usando Python y scikit‑learn.

## ¿Qué hace?
1. Descarga el dataset público **SMS Spam Collection (UCI)**.
2. Convierte texto a números con **TF‑IDF** (unigrams + bigrams).
3. Entrena **Multinomial Naive Bayes**.
4. Guarda un **Pipeline** (`spam_pipeline.joblib`) y permite predecir desde consola.

## Cómo instalar
```bash
pip install -r requirements.txt
```

## Entrenar el modelo

Ejecuta este comando para entrenar el modelo y guardar el archivo (`spam_pipeline.joblib`):

```bash
python train_model.py
```

## Predecir desde la consola:

1. **python predict.py** "Win a free iPhone now!"
2. **python predict.py** "Hey, wanna go to the movies later?"
3. **python predict.py** "Hola, mañana nos reunimos para el proyecto."

## Resultados obtenidos:

1. Accuracy: 97.8%

2. Precision (spam): 100%

3. Recall (spam): 83.9%

4. Matriz de confusión:

    [[1448,    0],
     [  36,  188]]

## Interpretación rapida:

El modelo es conservador:

1. Nunca confunde un mensaje normal (ham) con spam (precisión muy alta).

2. Pero se le escapan algunos mensajes spam (recall más bajo).

3. Accuracy global de 97.8%, lo que es un resultado muy sólido para un modelo sencillo.

## Estructura del proyecto: 

Proyecto uno/
├─ data/                  ## dataset descargado (SMSSpamCollection)
├─ spam_pipeline.joblib   ## modelo entrenado
├─ train_model.py         ## script para entrenar
├─ predict.py             ## script para predecir
├─ sms-classifier.py      ## script original (todo en uno)
├─ requirements.txt
└─ README.md


## Notas finales:

1. Proyecto realizado en Python 3.12.

2. Librerías principales: pandas, scikit-learn, joblib, requests.

3. Dataset: SMS Spam Collection - UCI Machine Learning Repository
