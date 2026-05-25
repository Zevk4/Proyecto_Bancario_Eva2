# README para Proyecto Bancario Eva2

## Descripción del Proyecto

Este proyecto es una solución analítica avanzada para predecir el comportamiento de abandono (churn) de clientes de una institución financiera.

## Requisitos del Sistema

### Versión de Python
- **Python 3.12** (requerido para reproducibilidad exacta) [2](#0-1) 

### Bibliotecas Necesarias
- **pandas** (>= 1.1.0) — Estructuración tabular y manipulación de datos
- **numpy** (>= 2.0.2) — Cómputo matricial y soporte algebraico
- **scikit-learn** (>= 1.3.0) — Construcción de Pipelines, Imputación Iterativa y algoritmos supervisados
- **matplotlib** (>= 3.7.1) — Motor gráfico para visualizaciones base
- **seaborn** (>= 0.13.2) — Estructuras de visualización estadística complementaria [3](#0-2) 

## Instalación Paso a Paso

### 1. Clonar u organizar el directorio de trabajo
Asegúrese de estar posicionado en la raíz de la carpeta descomprimida o clonada:

```bash
cd /ruta/hacia/tu/directorio/Proyecto_Bancario_Eva2
```

### 2. Creación y Activación del Entorno Virtual (venv)
Aísle las dependencias globales de Python creando un entorno local:

**En sistemas Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**En sistemas Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalación de dependencias
```bash
pip install -r requirements.txt
```

## Estructura de Carpetas

```
Proyecto_Bancario_Eva2/  
├── data/  
│   └── data_sucia/  
│       └── dataset_clientes.csv          # Dataset de clientes (datos crudos)  
├── docs/
│   └── Entrega_2_informe_Tecnico.pdf  
├── notebooks/  
│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio de datos  
│   ├── 02_supervised_modeling.ipynb     # Modelamiento supervisado y pipeline  
│   └── 03_model_evaluation.ipynb        # Evaluación comparativa de modelos 
│   └──04_hyperparameter_optimization.ipynb 
├── src/  
│   ├── data_preprocessing.py             # Funciones de preprocesamiento y feature engineering  
│   └── model_evaluation.py              # Funciones de evaluación de modelos  
├── results/  
│   └── metrics/  
│       └── analisis_umbrales_lr.csv     # Resultados de análisis de umbrales  
├── requirements.txt                      # Dependencias del proyecto  
└── README.md                             # Documentación del proyecto
```

## Descripción de Componentes

### notebooks/01_exploratory_analysis.ipynb
Notebook para análisis exploratorio que incluye la descarga del dataset desde GitHub y carga inicial de datos [4](#0-3) [5](#0-4) .

### notebooks/02_supervised_modeling.ipynb
Notebook principal que contiene el pipeline de modelamiento supervisado, incluyendo la configuración del entorno y las importaciones necesarias [6](#0-5) .

### src/data_preprocessing.py
Módulo que contiene funciones de preprocesamiento:
- `tratar_duplicados()` — Eliminación de registros duplicados [7](#0-6) 
- `corregir_signos_negativos()` — Corrección de valores negativos en variables financieras [8](#0-7) 
- `ingenieria_caracteristicas()` — Creación de variables derivadas para el modelo [9](#0-8) 

## Ejecución del Proyecto

### Paso 1: Análisis Exploratorio
1. Abra `notebooks/01_exploratory_analysis.ipynb` en Jupyter Notebook o JupyterLab
2. Ejecute las celdas secuencialmente para descargar el dataset y realizar el análisis inicial

### Paso 2: Modelamiento Supervisado
1. Abra `notebooks/02_supervised_modeling.ipynb`
2. Asegúrese de tener el dataset `dataset_clientes.csv` en el directorio raíz
3. Ejecute las celdas para:
   - Configurar el entorno e importar librerías
   - Aplicar el pipeline de preprocesamiento
   - Entrenar y evaluar los modelos de clasificación

## Verificación de Instalación

Para verificar que las librerías están correctamente instaladas, ejecute:

```python
import pandas as pd
print(pd.__version__)
```

Esto debería mostrar la versión instalada de pandas (ej: 3.0.3) [10](#0-9) .

## Notas
El proyecto utiliza un enfoque de feature engineering que genera variables financieras (ratio_deuda, ratio_gasto), métricas de comportamiento (indice_lealtad, riesgo_por_inactividad) e interacciones categóricas (region_uso_app) para mejorar la precisión del modelo de predicción de churn [9](#0-8) .

