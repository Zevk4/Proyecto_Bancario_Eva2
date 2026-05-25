import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Función para eliminar duplicados
def tratar_duplicados(X: pd.DataFrame, drop=True):
    """
    Tratamiento de duplicados.
    Parámetros:
      X : DataFrame
      drop : True (Si se deben eliminar los duplicados)
    Retorna: DataFrame limpio
    """
    return X.drop_duplicates() if drop else X

# 2. Función para corregir los signos negativos en las variables financieras
def corregir_signos_negativos(X: pd.DataFrame, columnas: list = None):
    """
    Transforma valores negativos a absolutos en columnas específicas.
    Parámetros:
      X : DataFrame
      columnas : list (Nombres de las columnas a transformar)
    Retorna: DataFrame con los signos corregidos
    """
    X_copy = X.copy()
    if columnas is not None:
        X_copy[columnas] = X_copy[columnas].abs()
    return X_copy

# 3. Función para la creación de variables (Feature Engineering)
def ingenieria_caracteristicas(X: pd.DataFrame, col_ingreso='ingreso_mensual', 
                               col_gasto='gasto_mensual', col_deuda='deuda_total',
                               col_antiguedad='antiguedad_meses', col_frecuencia='frecuencia_compra',
                               col_ultima_compra='ultima_compra_dias'):
    """
    Genera nuevas características financieras y de comportamiento.
    Implementa validación de columnas y Epsilon para estabilidad numérica.
    """
    X_copy = X.copy()
    eps = 1e-6  # Constante para evitar división por cero
    
    # Variables Financieras
    if all(col in X_copy.columns for col in [col_deuda, col_ingreso, col_gasto]):
        X_copy['ratio_deuda'] = X_copy[col_deuda] / (X_copy[col_ingreso] + eps)
        X_copy['ratio_gasto'] = X_copy[col_gasto] / (X_copy[col_ingreso] + eps)
    
    # Variables de Comportamiento
    if all(col in X_copy.columns for col in [col_antiguedad, col_frecuencia, col_ultima_compra]):
        X_copy['indice_lealtad'] = X_copy[col_antiguedad] / (X_copy[col_frecuencia] + eps)
        X_copy['riesgo_por_inactividad'] = X_copy[col_ultima_compra] / (X_copy[col_antiguedad] + eps)
        
    return X_copy

# 4. Clase Winsorizer para aplicar un recorte a los atípicos extremos
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Tratamiento de atípicos por truncamiento en percentiles extremos.
    Parámetros:
      limits : tuple (% de los extremos a descartar, por defecto 1% inferior y superior)
    """
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1 - self.limits[1])
            X = X.astype("float64")
            X[col] = np.clip(X[col], lower, upper)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.columns_)
        else:
            return np.array(input_features)