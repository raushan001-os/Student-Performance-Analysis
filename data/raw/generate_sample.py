"""
Generates a synthetic dataset that mirrors the Kaggle StudentsPerformance.csv schema.
Run this script once to create the sample data file used by the analysis.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

gender = rng.choice(["female", "male"], n, p=[0.52, 0.48])
race = rng.choice(
    ["group A", "group B", "group C", "group D", "group E"],
    n, p=[0.09, 0.19, 0.32, 0.26, 0.14]
)
parental_edu = rng.choice(
    ["some high school", "high school", "some college",
     "associate's degree", "bachelor's degree", "master's degree"],
    n, p=[0.18, 0.20, 0.23, 0.18, 0.13, 0.08]
)
lunch = rng.choice(["standard", "free/reduced"], n, p=[0.65, 0.35])
test_prep = rng.choice(["none", "completed"], n, p=[0.64, 0.36])

base_math  = rng.normal(65, 15, n)
base_read  = rng.normal(69, 14, n)
base_write = rng.normal(68, 15, n)

prep_boost  = np.where(test_prep == "completed", 6, 0)
lunch_boost = np.where(lunch == "standard", 5, 0)

edu_map = {
    "some high school": 0, "high school": 2, "some college": 4,
    "associate's degree": 6, "bachelor's degree": 8, "master's degree": 10
}
edu_boost = np.array([edu_map[e] for e in parental_edu])

math_score  = np.clip(base_math  + prep_boost + lunch_boost + edu_boost * 0.4 + rng.normal(0, 3, n), 0, 100).astype(int)
read_score  = np.clip(base_read  + prep_boost + lunch_boost + edu_boost * 0.5 + rng.normal(0, 3, n), 0, 100).astype(int)
write_score = np.clip(base_write + prep_boost + lunch_boost + edu_boost * 0.5 + rng.normal(0, 3, n), 0, 100).astype(int)

df = pd.DataFrame({
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": parental_edu,
    "lunch": lunch,
    "test preparation course": test_prep,
    "math score": math_score,
    "reading score": read_score,
    "writing score": write_score,
})

output_path = "StudentsPerformance.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} rows → {output_path}")
print(df.head())
