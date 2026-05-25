# src/model_evaluation.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluar_modelo_entrenado(modelo: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Calcula y retorna las métricas de rendimiento de un modelo clasificador ya entrenado.
    """
    # Generar predicciones directas y probabilidades sobre el set
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Diccionario de Metricas
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }