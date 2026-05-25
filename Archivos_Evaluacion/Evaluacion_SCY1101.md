# Evaluación Parcial N°2: Encargo con Presentación

## Información General de la Asignatura

| Sigla | Nombre Asignatura | Tiempo Asignado | % Ponderación |
| --- | --- | --- | --- |
| **SCY1101** | Programación para la Ciencia de Datos | 2 semanas | 30% |

---

## 1. Instrucciones Generales

Esta evaluación busca que el estudiante experimente el ciclo completo de desarrollo de una solución de **machine learning** orientada a un problema de negocio real. Cada equipo deberá integrar modelos supervisados, realizar una comparación rigurosa y optimización, y documentar su justificación técnica y análisis experimental.

### Algoritmos a Utilizar:

* **Regresión:** Linear Regression, Decision Tree Regressor.
* **Clasificación:** Logistic Regression, Decision Tree Classifier y SVM (Support Vector Machine).

El foco está en que los estudiantes seleccionen, implementen, evalúen y optimicen distintos modelos, aplicando las mejores prácticas de ingeniería de datos y machine learning. Además, se promueve el razonamiento crítico, la argumentación y la capacidad de comunicar resultados técnicos a través de informes y presentaciones.

### Distribución de Porcentajes de la Evaluación:

| Evaluación | Tipo de situación evaluativa | Distribución de porcentajes en el ET | Individual / Grupal |
| --- | --- | --- | --- |
| **Evaluación sumativa II** | Encargo | 10% | Grupal |
| **Evaluación Parcial N°2** | Presentación | 20% | Individual |
| **Total** |  | **30%** |  |

> **Nota sobre el tiempo:** El tiempo asignado para desarrollar esta evaluación en el laboratorio es de 2 semanas para el encargo y 15 minutos para la presentación. Se realiza en modalidades de forma individual, parejas o equipos.

---

## Instrucciones Específicas de la Evaluación

### Ítem I: Consideraciones para el Encargo (Grupal)

#### Focos de Observación:

* Organización y reproducibilidad del proyecto.
* Calidad técnica en la implementación de modelos y pipelines.
* Profundidad y rigor del análisis experimental.
* Justificación y claridad en la elección de algoritmos y técnicas.

#### El Entregable Debe Incluir:

##### 1. Estructura de Carpetas y Archivos

Se recomienda la siguiente organización en una carpeta raíz (`proyecto_modelado/`):

```text
proyecto_modelado/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── models/
│   └── trained_models/
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
└── README.md

```

* **Descripción de los Notebooks (`notebooks/`):**
* `01_exploratory_analysis.ipynb`: Análisis exploratorio de los datos, visualizaciones y detección de patrones.
* `02_supervised_modeling.ipynb`: Implementación de modelos supervisados (clasificación/regresión) con Scikit-learn.
* `03_model_evaluation.ipynb`: Evaluación comparativa, métricas, interpretación de resultados.
* `04_hyperparameter_optimization.ipynb`: Optimización de hiperparámetros (GridSearchCV / RandomizedSearchCV), documentación de resultados.
* `05_final_analysis.ipynb`: Integración y análisis final.


* **Descripción de los Módulos (`src/`):**
* `data_preprocessing.py`: Funciones de limpieza, transformación y preparación de datos.
* `model_training.py`: Definición y entrenamiento de modelos.
* `model_evaluation.py`: Funciones de evaluación y comparación.
* `hyperparameter_tuning.py`: Funciones para optimización de hiperparámetros.


* **Otros elementos:**
* `models/trained_models/`: Modelos entrenados serializados.
* `results/metrics/`, `plots/`, `reports/`: Métricas, visualizaciones y reportes.
* `README.md`: Guía de uso del proyecto y dependencias.



##### 2. Informe Técnico (12-15 páginas)

* **Resumen ejecutivo:** Descripción general, objetivo y resultados principales.
* **Marco metodológico:** Justificación de la selección de algoritmos y técnicas empleadas.
* **Análisis experimental:** Descripción detallada de experimentos, datos usados y configuraciones.
* **Resultados y comparación de modelos:** Reporte de métricas, tablas y gráficos comparativos.
* **Optimización de hiperparámetros:** Proceso seguido, justificación de los métodos y análisis del impacto.
* **Conclusiones y recomendaciones:** Reflexión sobre resultados, dificultades, alternativas evaluadas y recomendaciones futuras.
* **Referencias:** Bibliografía utilizada.

##### 3. Aspectos Formales del Código y Reproduducibilidad

* Código limpio, modular y documentado (incluyendo *docstrings*).
* Manejo robusto de excepciones y validación de entradas.
* Uso eficiente de recursos computacionales (uso de seeds, `random_state`, evitar procesamiento innecesario).
* El entregable debe ser **100% reproducible**.
* Uso de Scikit-learn, pandas, numpy, matplotlib/seaborn y buenas prácticas de ML.

---

### Ítem II: Consideraciones para la Presentación (Individual)

#### Focos de Observación:

* Dominio conceptual sobre selección, implementación, evaluación y optimización de modelos.
* Capacidad de explicar decisiones técnicas y metodológicas con argumentos claros y ejemplos concretos.
* Claridad al interpretar y comparar métricas de evaluación.
* Uso de visualizaciones efectivas y comunicación científica.

---

## 2. Pauta de Evaluación

### Descripción de Niveles de Logro

| Categoría | % Logro | Descripción |
| --- | --- | --- |
| **Muy buen desempeño** | 100% | Demuestra un desempeño destacado, evidenciando el logro de todos los aspectos evaluados en el indicador. |
| **Buen desempeño** | 80% | Demuestra un alto desempeño del indicador, presentando pequeñas omisiones, dificultades y/o errores. |
| **Desempeño aceptable** | 60% | Demuestra un desempeño competente, evidenciando el logro de los elementos básicos del indicador, pero con omisiones, dificultades o errores. |
| **Desempeño incipiente** | 30% | Presenta importantes omisiones, dificultades o errores en el desempeño, que no permiten evidenciar los elementos básicos del logro del indicador, por lo que no puede ser considerado competente. |
| **Desempeño no logrado** | 0% | Presenta ausencia o incorrecto desempeño. |

---

### Matriz de Evaluación: Dimensión Encargo (Grupal)

| Indicador de Evaluación | Muy Buen Desempeño (100%) | Buen Desempeño (80%) | Desempeño Aceptable (60%) | Desempeño Incipiente (30%) | Desempeño No Logrado (0%) | Ponderación |
| --- | --- | --- | --- | --- | --- | --- |
| **1. IEE 2.1.1:** Implementa múltiples modelos de clasificación y regresión usando Scikit-learn con configuraciones apropiadas y justificación técnica sólida. | Implementa y configura múltiples modelos con pipelines, justifica técnicamente cada decisión y demuestra dominio de Scikit-learn. | Implementa y configura modelos principales con pipelines básicos, buena justificación. | Implementa modelos básicos con configuración simple, justificación superficial. | Implementa algunos modelos con errores y sin justificación clara. | No implementa correctamente los modelos o el código no ejecuta. | **20%** |
| **2. IEE 2.2.1:** Evalúa modelos usando validación cruzada, implementando múltiples métricas de rendimiento e interpretando resultados comparativamente. | Aplica múltiples técnicas no supervisadas*, selecciona y evalúa con métricas y visualizaciones avanzadas. | Aplica técnicas principales y métricas básicas, visualización suficiente. | Aplica solo técnicas estándar, visualización simple. | Técnicas no supervisadas* incompletas o con errores. | No aplica técnicas o hay errores graves. | **40%** |
| **3. IEE 2.3.1:** Optimiza hiperparámetros usando GridSearchCV y RandomizedSearchCV, documentando el proceso y analizando el impacto en el rendimiento. | Implementa optimización exhaustiva y justifica técnica y visualmente el proceso y sus efectos. | Optimización adecuada con justificación suficiente. | Optimización básica, justificación elemental. | Optimización incompleta o sin justificación. | No realiza optimización o hay errores graves. | **40%** |

**Nota: El texto original del documento indica explícitamente "técnicas no supervisadas" en las celdas descriptivas del indicador IEE 2.2.1.*

---

### Matriz de Evaluación: Dimensión Presentación (Individual)

| Indicador de Evaluación | Muy Buen Desempeño (100%) | Buen Desempeño (80%) | Desempeño Aceptable (60%) | Desempeño Incipiente (30%) | Desempeño No Logrado (0%) | Ponderación |
| --- | --- | --- | --- | --- | --- | --- |
| **5. IEP 2.1.3:** Explica la implementación de modelos supervisados, justificando la selección de algoritmos según la naturaleza del problema. | Explica y justifica todas las decisiones técnicas con argumentos sólidos y ejemplos claros. | Explica bien la mayoría de las decisiones, con justificación suficiente. | Explica lo esencial, justificación superficial. | Explicación incompleta o poco técnica. | No explica o es incorrecta. | **30%** |
| **6. IEP 2.2.2:** Interpreta y compara métricas de evaluación, explicando el significado de cada métrica en el contexto del problema específico. | Interpreta todas las métricas relevantes, compara modelos y explica tradeoffs con visualizaciones avanzadas. | Interpreta métricas principales y compara modelos, explicación suficiente. | Interpreta algunas métricas, comparación superficial. | Interpretación incompleta, sin visualización adecuada. | No interpreta o lo hace incorrectamente. | **35%** |
| **7. IEP 2.3.2:** Explica el proceso de optimización de hiperparámetros y su impacto en el rendimiento final de los modelos. | Explica el proceso completo, describe y analiza impacto con claridad y propuestas de mejora. | Explica proceso y análisis principal, conclusiones suficientes. | Explicación básica del proceso, análisis superficial. | Explicación incompleta o errónea. | No explica o es incorrecto. | **35%** |

**Total Dimensión Presentación:** 100%

---

*Metadatos del Documento Institucional: MALETA DIDÁCTICA Duoc UC - Subdirección de Diseño Instruccional - Año 2025.*