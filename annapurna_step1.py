# Annapurna 2.0 – Step 1: Dataset Creation, Libraries and Exploratory Data Analysis
# Practicals covered:
# 1. Introduction to Python ML libraries
# 2. Dataset creation and loading
# 3. Exploratory Data Analysis (EDA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "annapurna_data.csv"


# ============================================================
# PRACTICAL 1: INTRODUCTION TO ML LIBRARIES
# ============================================================

print("\n========== PRACTICAL 1: ML LIBRARIES ==========")

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)

# NumPy demonstration
numbers = np.array([10, 20, 30, 40, 50])

print("\nNumPy Array:", numbers)
print("Mean:", np.mean(numbers))
print("Standard Deviation:", np.std(numbers))

# Pandas demonstration
sample_df = pd.DataFrame({
    "Student": ["A", "B", "C", "D"],
    "Marks": [85, 90, 78, 92]
})

print("\nPandas DataFrame:")
print(sample_df)

print("\nPandas Statistics:")
print(sample_df.describe())

# Matplotlib demonstration
plt.figure(figsize=(8, 5))
plt.plot(numbers, marker="o")
plt.title("Simple NumPy and Matplotlib Demonstration")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)

plt.savefig(OUTPUT_DIR / "p1_matplotlib_demo.png")
plt.close()

print("\nPractical 1 completed.")


# ============================================================
# PRACTICAL 2: DATASET CREATION AND LOADING
# ============================================================

print("\n========== PRACTICAL 2: DATASET CREATION ==========")


def make_sample_data(n_days=60, seed=42):
    """
    Generate simulated college mess data.

    Each day contains three meal records:
    Breakfast, Lunch and Dinner.

    IMPORTANT:
    This is simulated educational data, not actual college mess data.
    """

    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        start="2026-09-01",
        periods=n_days,
        freq="D"
    )

    meals = ["Breakfast", "Lunch", "Dinner"]

    menu_options = {
        "Breakfast": ["Poha", "Upma", "Idli", "Paratha"],
        "Lunch": ["Dal Rice", "Paneer Curry", "Chole", "Rajma"],
        "Dinner": ["Khichdi", "Veg Biryani", "Roti Sabzi", "Pav Bhaji"]
    }

    rows = []

    for date in dates:

        for meal in meals:

            day_of_week = date.day_name()

            exam_week = rng.choice([0, 1], p=[0.8, 0.2])
            holiday_or_event = rng.choice([0, 1], p=[0.9, 0.1])
            rain = rng.choice([0, 1], p=[0.7, 0.3])

            temperature_c = rng.normal(29, 4)

            students_registered = int(rng.integers(180, 251))

            # Expected attendance changes based on day and events.
            attendance_rate = 0.88

            if day_of_week == "Sunday":
                attendance_rate -= 0.15

            if holiday_or_event == 1:
                attendance_rate -= 0.20

            if exam_week == 1:
                attendance_rate -= 0.05

            students_expected = int(
                students_registered * attendance_rate
            )

            plates_cooked = int(
                students_expected * rng.uniform(1.02, 1.12)
            )

            plates_eaten = int(
                students_expected * rng.uniform(0.85, 1.05)
            )

            plates_eaten = min(plates_eaten, plates_cooked)

            leftover_kg = max(
                0,
                (plates_cooked - plates_eaten)
                * rng.uniform(0.12, 0.20)
            )

            menu_main = rng.choice(menu_options[meal])

            rows.append({
                "date": date,
                "day_of_week": day_of_week,
                "meal": meal,
                "menu_main": menu_main,
                "exam_week": exam_week,
                "holiday_or_event": holiday_or_event,
                "rain": rain,
                "temperature_c": round(temperature_c, 2),
                "students_registered": students_registered,
                "plates_cooked": plates_cooked,
                "plates_eaten": plates_eaten,
                "leftover_kg": round(leftover_kg, 2),
                "data_source": "simulated"
            })

    df = pd.DataFrame(rows)

    # Add a few missing values for the data-cleaning practical.
    missing_temp_indices = rng.choice(
        df.index,
        size=4,
        replace=False
    )

    missing_eaten_indices = rng.choice(
        df.index,
        size=4,
        replace=False
    )

    df.loc[missing_temp_indices, "temperature_c"] = np.nan
    df.loc[missing_eaten_indices, "plates_eaten"] = np.nan

    # Define high food-waste risk using the 70th percentile.
    waste_threshold = df["leftover_kg"].quantile(0.70)

    df["high_waste"] = (
        df["leftover_kg"] >= waste_threshold
    ).astype(int)

    return df


# Generate the educational sample dataset.
df = make_sample_data()

# Save only if the file does not already exist.
# This prevents accidentally overwriting collected real data.
if DATA_FILE.exists():
    print("Existing dataset found. It will NOT be overwritten.")
    df = pd.read_csv(DATA_FILE)
else:
    df.to_csv(DATA_FILE, index=False)
    print("New simulated dataset saved successfully.")

# Display dataset information.
print("\nDataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

# Identify features and targets.
target_regression = "plates_eaten"
target_classification = "high_waste"

excluded_columns = [
    "date",
    "data_source",
    "plates_eaten",
    "high_waste",
    "leftover_kg"
]

features = [
    column for column in df.columns
    if column not in excluded_columns
]

print("\nInput features:")
print(features)

print("\nRegression target:", target_regression)
print("Classification target:", target_classification)

numeric_features = df[features].select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = df[features].select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

print("\nPractical 2 completed.")


# ============================================================
# PRACTICAL 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n========== PRACTICAL 3: EDA ==========")

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nHigh Waste Class Distribution:")
print(df["high_waste"].value_counts())

print("\nAverage Leftover by Meal:")
print(df.groupby("meal")["leftover_kg"].mean())

print("\nAverage Leftover by Day:")
print(df.groupby("day_of_week")["leftover_kg"].mean())


# Plot 1: Distribution of plates eaten.
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="plates_eaten",
    bins=20,
    kde=True
)

plt.title("Distribution of Plates Eaten")
plt.xlabel("Plates Eaten")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "p3_plates_eaten_distribution.png")
plt.close()


# Plot 2: Average leftover by date.
daily_leftover = df.groupby("date")["leftover_kg"].mean()

plt.figure(figsize=(12, 5))

plt.plot(
    daily_leftover.index,
    daily_leftover.values,
    marker="o"
)

plt.title("Average Food Leftover by Date")
plt.xlabel("Date")
plt.ylabel("Average Leftover (kg)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "p3_daily_leftover.png")
plt.close()


# Plot 3: Leftover by meal and exam week.
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="meal",
    y="leftover_kg",
    hue="exam_week"
)

plt.title("Food Leftover by Meal and Exam Week")
plt.xlabel("Meal")
plt.ylabel("Leftover (kg)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "p3_leftover_by_meal.png")
plt.close()


# Plot 4: Numeric correlation heatmap.
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numeric Variables")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "p3_correlation_heatmap.png")
plt.close()


print("\nEDA plots saved in:", OUTPUT_DIR)

print("\n========== STEP 1 COMPLETED ==========")
print("Dataset and EDA outputs are ready.")
print("Remember: the generated dataset is simulated, not real mess data.")
