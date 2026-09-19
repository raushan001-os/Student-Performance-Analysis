# Student Performance Analysis

Exploratory analysis and predictive modeling on the **Students Performance in Exams** dataset from Kaggle.
Developed as a technical demo for the Data Mining / Intelligent Systems course.

---

## Objective

Identify the factors that most influence academic performance and build a model capable of predicting whether a student will pass (average score ≥ 60) based on socio-educational variables.

---

## Dataset

**Kaggle — Students Performance in Exams**
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

| Variable | Type | Description |
|---|---|---|
| gender | Categorical | Student's gender |
| race/ethnicity | Categorical | Ethnic group (A–E) |
| parental level of education | Ordinal | Parent's education level |
| lunch | Categorical | Lunch type (standard / subsidized) |
| test preparation course | Categorical | Whether the student completed the preparation course |
| math score | Numerical | Mathematics score (0–100) |
| reading score | Numerical | Reading score (0–100) |
| writing score | Numerical | Writing score (0–100) |

Download the CSV and place it in `data/raw/StudentsPerformance.csv`.

---

## Project Structure

```text
student-performance-analysis/
├── src/
│   ├── eda.py          # Exploratory data analysis + 6 figures
│   ├── features.py     # Feature engineering and preprocessing
│   ├── model.py        # Decision tree + metrics + visualizations
│   └── analysis.py     # Main runner (orchestrates the 3 modules)
├── data/
│   └── raw/
│       └── StudentsPerformance.csv   ← place the dataset here
├── outputs/
│   ├── figures/        # 9 automatically generated charts
│   ├── metrics.json    # Model metrics in JSON format
│   └── tree_rules.txt  # Decision tree rules in plain text
├── powerbi/
│   ├── dashboard-screenshots/
│   ├── Student_Performance_PowerBI_Ready.xlsx
│   ├── Student_Performance_Project_Package.xlsx
│   └── README.md
├── requirements.txt
└── README.md

---

## Licencia

MIT — libre para uso académico y proyectos personales.
