"""
Survey augmentation.

Takes the cleaned survey export and:
  - adds answers for the new questions Q40-Q46 to every existing respondent
    (conditioned on income S7, age S2 and education S3)
  - tops the dataset up to 10,000 rows with synthetic respondents, sampling each
    answer from the distribution observed in the real data

Usage:
    python src/augment_survey.py data/clean03-05_needingadd.xlsx -o data/augmented_dataset.xlsx
"""
import argparse

import numpy as np
import pandas as pd

TARGET_ROWS = 10_000

# New questions added to the questionnaire (codes -> labels, kept for reference)
NEW_QUESTIONS = {
    "Q40": {1: "Price", 2: "Results", 3: "Flexibility", 4: "Support"},
    "Q41": {1: "Saving money", 2: "Getting the best performance",
            3: "Staying connected without interruption"},
    "Q42": {1: "Very informed", 2: "Somewhat informed", 3: "Not informed"},
    "Q43": {1: "Always", 2: "Sometimes", 3: "Never"},
    "Q44": {1: "Yes", 2: "No"},
    "Q45": {1: "Phone Calls", 2: "SMS/Text Messages", 3: "Email",
            4: "Social Media", 5: "In-Person Visits"},
    "Q46": {1: "Totally Unsatisfied", 2: "Unsatisfied", 3: "Satisfied",
            4: "Highly Satisfied", 5: "NSP"},
}

YES_NO = [1, 2]
BINARY = [0, 1]
LIKERT5 = [1, 2, 3, 4, 5]

# Columns sampled from the empirical distribution of the real answers
EMPIRICAL = {
    "DISTRICT": ["Centre Est", "Sud Ouest", "Grand Tunis", "Nord Ouest"],
    "GOUVERNORAT": ["Sousse", "Gafsa", "Ben Arous", "Siliana", "Kebili"],
    "DELEGATION": ["Sidi El Heni", "Oum El Araies", "Hammam Lif", "El Krib", "Souk El Ahed"],
    "I0": [1, 2, 3],
    "S1": YES_NO,
    "S2": list(range(1, 10)),
    "S3": list(range(1, 6)),
    "S7": list(range(1, 8)),
    "S8": [1, 2, 3, 4],
    "Q1": YES_NO,
    "Q2_3": [1, 2, 3, 4],
    "Q3": YES_NO,
    "Q5": [1, 2, 3],
    "Q6_1": YES_NO, "Q6_2": YES_NO, "Q6_3": YES_NO, "Q6_4": YES_NO,
    "Q8_3": [1, 2, 7, 3, 4, 8, 5, 6, 9],
    "Q8_4": BINARY,
    "Q15": YES_NO,
    **{f"Q16_{i}": YES_NO for i in range(1, 10)},
    **{f"Q16_{i}": BINARY for i in range(10, 13)},
    "Q17": list(range(1, 7)),
    "Q18": list(range(1, 7)),
    "Q19": [1, 2, 3],
    **{f"Q20_{i}": BINARY for i in range(1, 13)},
    "Q20BIS": YES_NO,
    "Q26A": YES_NO,
    "Q26": [1, 2, 3, 4],
    "Q27": LIKERT5,
    "Q28": [1, 2, 3, 4],
    "Q29": [1, 2, 98],
    "Q30": [1, 2, 3],
    "Q34": YES_NO,
    "Q35": [1, 2, 3],
    "Q36": [1, 2, 3, 4],
    "Q36_1": YES_NO,
    **{f"Q37_{i}": LIKERT5 for i in range(1, 12)},
}

# Columns with no useful distribution in the source: plain uniform draws
# (low, high) for randint, high is exclusive
UNIFORM_INT = {
    "ID_aud": (1000, 9999), "GOUV": (1, 30), "test6": (1, 100), "S4": (1, 10),
    "S6_1_B": (1, 20), "S6_2_B": (1, 20), "Q2": (1, 6), "Q7": (1, 5),
    "Q8": (1, 6), "test7": (1, 50), "Q31val": (100, 2000),
}
UNIFORM_FLOAT = {
    "LATITUDE1": (30, 37), "LONGITUDE1": (7, 12),
    "Q32val": (1, 50), "Q33val": (10, 100),
}

# Fallback probabilities for Q43-Q46 if the source has no answers yet
FALLBACK = {
    "Q43": [0.1, 0.3, 0.6],
    "Q44": [0.4, 0.6],
    "Q45": [0.3, 0.2, 0.2, 0.2, 0.1],
    "Q46": [0.1, 0.2, 0.4, 0.2, 0.1],
}


def empirical_probs(df, column, values):
    """Share of each value in `column`; unseen values get a small floor."""
    if column not in df.columns:
        return np.full(len(values), 1 / len(values))
    counts = df[column].value_counts(normalize=True)
    p = np.array([counts.get(v, 0.01) for v in values], dtype=float)
    return p / p.sum()


def add_new_questions(row, df, rng):
    """Q40-Q42 depend on the respondent's profile, Q43-Q46 follow the data."""
    row["Q40"] = rng.choice([1, 2, 3, 4],
                            p=[0.6, 0.2, 0.1, 0.1] if row.get("S7", 4) <= 4 else [0.2, 0.6, 0.1, 0.1])
    row["Q41"] = rng.choice([1, 2, 3],
                            p=[0.6, 0.3, 0.1] if row.get("S2", 4) <= 4 else [0.2, 0.3, 0.5])
    row["Q42"] = rng.choice([1, 2, 3],
                            p=[0.5, 0.4, 0.1] if row.get("S3", 4) >= 4 else [0.2, 0.4, 0.4])
    for q, fallback in FALLBACK.items():
        values = list(NEW_QUESTIONS[q])
        p = empirical_probs(df, q, values) if q in df.columns else fallback
        row[q] = rng.choice(values, p=p)
    return row


def synthetic_respondent(df, rng):
    row = {}
    for col, (lo, hi) in UNIFORM_INT.items():
        row[col] = rng.integers(lo, hi)
    for col, (lo, hi) in UNIFORM_FLOAT.items():
        row[col] = rng.uniform(lo, hi)
    for col, values in EMPIRICAL.items():
        row[col] = rng.choice(values, p=empirical_probs(df, col, values))
    return add_new_questions(row, df, rng)


def augment(df, target=TARGET_ROWS, seed=42):
    rng = np.random.default_rng(seed)
    df = df.apply(lambda r: add_new_questions(r.copy(), df, rng), axis=1)

    missing = target - len(df)
    if missing <= 0:
        print(f"Dataset already has {len(df)} rows, nothing to add.")
        return df

    synth = pd.DataFrame([synthetic_respondent(df, rng) for _ in range(missing)])
    synth = synth.reindex(columns=df.columns)       # same schema as the real data
    out = pd.concat([df, synth], ignore_index=True)

    if out.isnull().any().any():
        print("Filling remaining NaNs with the column mode.")
        out = out.fillna(out.mode().iloc[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="cleaned survey (.xlsx)")
    ap.add_argument("-o", "--output", default="augmented_dataset.xlsx")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--rows", type=int, default=TARGET_ROWS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print("Loaded:", df.shape)
    out = augment(df, target=args.rows, seed=args.seed)
    print("Augmented:", out.shape)
    out.to_excel(args.output, index=False)
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
