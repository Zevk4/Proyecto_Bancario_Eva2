# src/model_evaluation.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

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

def graficar_comparacion_metricas(metricas_base: dict, metricas_optimizadas: dict, nombre_modelo: str):
    """
    Genera un gráfico de barras agrupadas comparando las métricas de un modelo antes y después de la optimización.
    
    Parámetros:
      metricas_base : dict (Métricas del modelo original devueltas por evaluar_modelo_entrenado)
      metricas_optimizadas : dict (Métricas del modelo optimizado devueltas por evaluar_modelo_entrenado)
      nombre_modelo : str (Nombre del modelo para el título, ej: 'Regresión Logística')
    """
    # Convertir los diccionarios a un DataFrame para facilitar la graficación con Seaborn
    df_metricas = pd.DataFrame([metricas_base, metricas_optimizadas], index=['Base', 'Optimizado']).T
    df_metricas.reset_index(inplace=True)
    df_metricas.rename(columns={'index': 'Métrica'}, inplace=True)
    
    # Formato largo (melt) necesario para gráficos agrupados en Seaborn
    df_melted = df_metricas.melt(id_vars='Métrica', var_name='Estado', value_name='Puntuación')
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Métrica', y='Puntuación', hue='Estado', data=df_melted, palette='mako')
    
    # Configuraciones estéticas
    plt.title(f'Comparación de Rendimiento: {nombre_modelo} (Base vs Optimizado)', fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    plt.ylabel('Puntuación', fontsize=12)
    plt.xlabel('Métrica', fontsize=12)
    
    # Agregar los valores numéricos sobre cada barra
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    
    plt.legend(loc='lower right', title='Versión del Modelo')
    plt.tight_layout()
    plt.show()

def graficar_curva_roc_comparativa(y_test, y_prob_base, y_prob_opt, nombre_modelo: str):
    """
    Grafica la curva ROC comparando el modelo base vs optimizado.
    """
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    roc_auc_base = auc(fpr_base, tpr_base)

    fpr_opt, tpr_opt, _ = roc_curve(y_test, y_prob_opt)
    roc_auc_opt = auc(fpr_opt, tpr_opt)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_base, tpr_base, color='royalblue', lw=2, linestyle='--', 
             label=f'Base (AUC = {roc_auc_base:.3f})')
    plt.plot(fpr_opt, tpr_opt, color='darkorange', lw=2, 
             label=f'Optimizado (AUC = {roc_auc_opt:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Comparación Curva ROC - {nombre_modelo}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_comparacion_matrices(y_test, y_pred_base, y_pred_opt, nombre_modelo: str):
    """
    Grafica dos matrices de confusión lado a lado para comparar el modelo base con el optimizado.
    """
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_opt = confusion_matrix(y_test, y_pred_opt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Matriz de Confusión - {nombre_modelo}\n(Modelo Base)')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Valor Real')

    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'Matriz de Confusión - {nombre_modelo}\n(Modelo Optimizado)')
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Valor Real')

    plt.tight_layout()
    plt.show()