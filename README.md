# Análisis de Rendimiento Estudiantil

Análisis exploratorio y modelo predictivo sobre el dataset **Students Performance in Exams** de Kaggle.
Desarrollado como demo técnica para la asignatura de Minería de Datos / Sistemas Inteligentes.

---

## Objetivo

Identificar los factores que más influyen en el rendimiento académico y construir un modelo capaz de predecir si un estudiante aprobará (promedio ≥ 60) con base en variables socioeducativas.

---

## Dataset

**Kaggle — Students Performance in Exams**
[https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

| Variable | Tipo | Descripción |
|---|---|---|
| gender | Categórica | Género del estudiante |
| race/ethnicity | Categórica | Grupo étnico (A–E) |
| parental level of education | Ordinal | Nivel educativo de los padres |
| lunch | Categórica | Tipo de almuerzo (estándar / subsidiado) |
| test preparation course | Categórica | Completó curso preparatorio (sí / no) |
| math score | Numérica | Puntaje en matemáticas (0–100) |
| reading score | Numérica | Puntaje en lectura (0–100) |
| writing score | Numérica | Puntaje en escritura (0–100) |

Descarga el CSV y colócalo en `data/raw/StudentsPerformance.csv`.

---

## Estructura del proyecto

```
student-performance-analysis/
├── src/
│   ├── eda.py          # Análisis exploratorio + 6 figuras
│   ├── features.py     # Ingeniería de variables y preprocesamiento
│   ├── model.py        # Árbol de decisión + métricas + visualizaciones
│   └── analysis.py     # Runner principal (orquesta los 3 módulos)
├── data/
│   └── raw/
│       └── StudentsPerformance.csv   ← coloca aquí el dataset
├── outputs/
│   ├── figures/        # 9 gráficas generadas automáticamente
│   ├── metrics.json    # Métricas del modelo en JSON
│   └── tree_rules.txt  # Reglas del árbol en texto plano
├── requirements.txt
└── README.md
```

---

## Cómo ejecutar

### 1. Instalar dependencias

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Colocar el dataset

```
data/raw/StudentsPerformance.csv
```

### 3. Ejecutar el análisis completo

```bash
python src/analysis.py
```

Salida esperada:

```
10:15:03 | INFO     | STUDENT PERFORMANCE ANALYSIS — STARTED
10:15:03 | INFO     | [1/4] Loading dataset...
10:15:03 | INFO     |       1,000 rows × 8 columns loaded
10:15:03 | INFO     | [2/4] Running EDA...
10:15:04 | INFO     | EDA complete. All figures saved to outputs/figures/
10:15:04 | INFO     | [3/4] Feature engineering & preprocessing...
10:15:04 | INFO     | Train size: 800   Test size: 200
10:15:04 | INFO     | [4/4] Training and evaluating Decision Tree...

── Model Evaluation ──────────────────────────────
  Accuracy : 0.7200
  ROC-AUC  : 0.7970

10:15:04 | INFO     | DONE — Accuracy: 0.7200  |  ROC-AUC: 0.7970
```

### 4. Explorar con un árbol más profundo

```bash
python src/analysis.py --depth 6
```

---

## Figuras generadas

| Archivo | Contenido |
|---|---|
| `01_score_distributions.png` | Histogramas de los tres puntajes con media marcada |
| `02_scores_by_gender.png` | Boxplots por género |
| `03_scores_by_lunch.png` | Boxplots por tipo de almuerzo |
| `04_scores_by_test_prep.png` | Boxplots por curso preparatorio |
| `05_parental_edu_heatmap.png` | Mapa de calor: puntaje medio según educación de los padres |
| `06_correlation_matrix.png` | Matriz de correlación de Pearson |
| `07_confusion_matrix.png` | Matriz de confusión del modelo |
| `08_feature_importances.png` | Importancia de variables (Gini) |
| `09_decision_tree.png` | Visualización completa del árbol de decisión |

---

## Resultados principales

### Variables con mayor influencia

El árbol de decisión identifica que las variables con mayor poder predictivo son:

1. **Tipo de almuerzo** — los estudiantes con almuerzo estándar obtienen consistentemente puntajes más altos (~8 puntos de diferencia en promedio)
2. **Curso preparatorio** — completarlo mejora el promedio en ~6 puntos
3. **Nivel educativo de los padres** — correlación positiva clara: a mayor educación parental, mayor rendimiento del estudiante

### Tasas de aprobación (promedio ≥ 60)

| Asignatura | Tasa |
|---|---|
| Lectura | 86.7 % |
| Escritura | 83.5 % |
| Matemáticas | 79.3 % |

Matemáticas presenta la mayor dispersión y la tasa de reprobación más alta — dato relevante para diseño curricular.

---

## Flujo del análisis

```
CSV raw
  │
  ├─ eda.py
  │    ├─ Estadísticas descriptivas
  │    ├─ Distribución de puntajes
  │    ├─ Comparativas por grupos (género, almuerzo, preparación)
  │    └─ Mapa de calor + correlaciones
  │
  ├─ features.py
  │    ├─ Derivación: average_score, passed, performance
  │    ├─ OrdinalEncoder (educación parental)
  │    ├─ OneHotEncoder (género, raza, almuerzo, preparación)
  │    └─ Train/test split estratificado (80/20)
  │
  └─ model.py
       ├─ DecisionTreeClassifier (max_depth=4)
       ├─ Accuracy + ROC-AUC
       ├─ Reporte por clase (precision / recall / F1)
       └─ Exportación de reglas en texto plano
```

---

## Ideas para extender este análisis

- Agregar **Random Forest** o **XGBoost** y comparar métricas con el árbol de decisión
- Aplicar **validación cruzada** (k-fold) para una evaluación más robusta
- Construir un **dashboard interactivo** con Streamlit sobre los resultados
- Probar con el dataset de **deserción estudiantil** de la UCI para un problema de mayor complejidad
- Incorporar **SHAP values** para explicabilidad avanzada del modelo

---

## Referencias

- Cortez, P. & Silva, A. (2008). *Using data mining to predict secondary school student performance*. EUROSIS.
- Breiman, L. et al. (1984). *Classification and Regression Trees*. Chapman & Hall.
- Scikit-learn documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

---

## Licencia

MIT — libre para uso académico y proyectos personales.
