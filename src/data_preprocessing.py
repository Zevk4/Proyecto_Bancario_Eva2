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
                               col_ultima_compra='ultima_compra_dias', col_region='region', 
                               col_uso_app='uso_app'):
    """
    Genera nuevas características financieras, de comportamiento e interacciones categóricas.
    """
    X_copy = X.copy()
    eps = 1e-6  # Constante para evitar división por cero (0.000001)

    # 1. Variables Financieras
    # Verificamos si las columnas existen en el dataset actual antes de operar
    if all(col in X_copy.columns for col in [col_deuda, col_ingreso, col_gasto]):
        X_copy['ratio_deuda'] = X_copy[col_deuda] / (X_copy[col_ingreso] + eps)
        X_copy['ratio_gasto'] = X_copy[col_gasto] / (X_copy[col_ingreso] + eps)

    # 2. Variables de Comportamiento
    if all(col in X_copy.columns for col in [col_antiguedad, col_frecuencia, col_ultima_compra]):
        X_copy['indice_lealtad'] = X_copy[col_antiguedad] / (X_copy[col_frecuencia] + eps)
        X_copy['riesgo_por_inactividad'] = X_copy[col_ultima_compra] / (X_copy[col_antiguedad] + eps)

    # 3. Interacción Categórica: Región por Uso de la App
    if all(col in X_copy.columns for col in [col_region, col_uso_app]):
        X_copy['region_uso_app'] = X_copy[col_region].astype(str) + '_' + X_copy[col_uso_app].astype(str)

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
        
#  5. Convierte un array en un DataFrame
class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Clase optimizada para conservar nombres de columnas, compatible con GridSearchCV."""
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Si el paso anterior ya entrega un DataFrame, extraemos las columnas directamente
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
        else:
            # Fallback de seguridad para ejecuciones directas sin configuración de salida
            if self.preprocessor is not None:
                self.feature_names_ = self.preprocessor.get_feature_names_out()
            else:
                self.feature_names_ = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names_)
    
# 6. Eliminación de variables correlacionadas
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Clase genérica para mitigar multicolinealidad eliminando variables con corr > threshold."""
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.columns_to_drop_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.columns_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_filtered = X_df.drop(columns=self.columns_to_drop_, errors="ignore")
        return X_filtered.values