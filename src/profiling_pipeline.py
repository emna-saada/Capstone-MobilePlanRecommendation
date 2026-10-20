"""
Customer profiling and mobile plan recommendation.

Steps:
  1. preprocessing (imputation, encoding, outlier clipping, engineered scores)
  2. feature selection with a Random Forest on provisional K-Means labels
  3. K-Means (k=4) to build the customer profiles
  4. compare LogReg / RF / XGBoost / KNN, tune XGBoost with a grid search
  5. SHAP explanations, export the model, map profiles to plan offers

Usage:
    python src/profiling_pipeline.py data/augmented_dataset.xlsx --out results
"""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_PROFILES = 4

CATEGORICAL = [
    "I0", "S1", "S2", "S3", "S6_1_B", "S6_2_B", "S7", "S8", "Q1", "Q2_3", "Q3", "Q5",
    "Q6_1", "Q6_2", "Q6_3", "Q6_4", "Q7", "Q15", *[f"Q16_{i}" for i in range(1, 9)],
    "Q17", "Q18", "Q19", "Q20BIS", "Q26A", "Q26", "Q27", "Q28", "Q29", "Q30", "Q34",
    "Q35", "Q36", "Q36_1", *[f"Q37_{i}" for i in range(1, 12)],
    "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46",
]
NUMERICAL = ["Q31val", "Q32val", "Q33val", "Q2", "Q8"]
NOMINAL = ["I0", "S1", "S6_1_B", "S6_2_B", "S8", "Q2_3", "Q26"]   # one-hot
ORDINAL = [
    "S2", "S3", "S7", "Q3", "Q5", "Q6_1", "Q6_2", "Q6_3", "Q6_4", "Q7", "Q15",
    "Q17", "Q18", "Q19", "Q20BIS", "Q26A", "Q27", "Q28", "Q29", "Q30", "Q34",
    "Q35", "Q36", "Q36_1", *[f"Q37_{i}" for i in range(1, 12)],
    "Q40", "Q41", "Q42", "Q43", "Q44", "Q45", "Q46",
]

# Cluster ids -> names. Assigned after looking at the profile averages
# (satisfaction, MB activated, income S7, usage Q19), so re-check them if
# the data or the random seed changes.
PROFILE_NAMES = {0: "Budget-Conscious", 1: "Heavy Data", 2: "Premium", 3: "Dissatisfied"}

OFFERS = {
    "Budget-Conscious": "Prepaid plan, 500 MB, 5 dt/month",
    "Heavy Data": "Postpaid, 10 GB, 20 dt/month, free social media",
    "Premium": "Unlimited data, priority support, 50 dt/month",
    "Dissatisfied": "Discounted plan, enhanced customer service, 15 dt/month",
}


# ---------------------------------------------------------------- preprocessing

def preprocess(df):
    df = df.copy()

    for col in CATEGORICAL:
        if col in df and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in NUMERICAL:
        if col in df and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # drop columns that are more than half empty
    df = df.dropna(axis=1, thresh=int(0.5 * len(df)))

    if all(c in df for c in NOMINAL):
        df = pd.get_dummies(df, columns=NOMINAL, prefix=NOMINAL)

    for col in ORDINAL:
        if col in df:
            df[col] = df[col].astype(int)

    # numeric answers: coerce, fill, clip to 5th-95th percentile
    for col in NUMERICAL:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].clip(df[col].quantile(0.05), df[col].quantile(0.95))

    # engineered scores
    sat_cols = [c for c in df if c.startswith("Q37_") or c == "Q46"]
    if sat_cols:
        df["Satisfaction_Score"] = df[sat_cols].mean(axis=1)
    if {"Q7", "Q8"} <= set(df.columns):
        df["Device_Intensity"] = df["Q7"] + df["Q8"]
    if {"Q18", "Q19"} <= set(df.columns):
        df["Internet_Usage_Score"] = df["Q18"] * df["Q19"]
    if "Q31val" in df:
        bins = pd.cut(df["Q31val"], bins=[0, 500, 2000, float("inf")],
                      labels=["Low", "Medium", "High"])
        df = pd.concat([df, pd.get_dummies(bins, prefix="Q31val")], axis=1)

    scale_cols = ["Q31val", "Q32val", "Q33val", "Satisfaction_Score",
                  "Device_Intensity", "Internet_Usage_Score", *NUMERICAL]
    scale_cols = sorted({c for c in scale_cols
                         if c in df and df[c].dtype in ("float64", "int64")})
    if scale_cols:
        df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

    # drop one of each pair with |corr| > 0.8 (in practice Device_Intensity)
    if len(scale_cols) > 1:
        corr = df[scale_cols].corr().abs()
        to_drop = {corr.columns[i] for i in range(len(corr))
                   for j in range(i) if corr.iloc[i, j] > 0.8}
        df = df.drop(columns=list(to_drop))
        print("Dropped correlated columns:", to_drop or "none")

    return df


# ---------------------------------------------------------- features + profiles

def select_features(df, n=20):
    """Top-n features from a Random Forest trained on provisional K-Means labels."""
    X = df.select_dtypes(include=[np.number, "bool"])
    tmp = KMeans(n_clusters=N_PROFILES, random_state=SEED).fit_predict(X)
    rf = RandomForestClassifier(random_state=SEED).fit(X, tmp)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:\n", imp.head(n))
    return imp.head(n).index.tolist()


def elbow_plot(X, out):
    ks = range(2, 10)
    inertia = [KMeans(n_clusters=k, random_state=SEED).fit(X).inertia_ for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertia, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow method")
    plt.tight_layout()
    plt.savefig(out / "elbow.png", dpi=150)
    plt.close()


# ------------------------------------------------------------------- modelling

def compare_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(random_state=SEED),
        "KNN": KNeighborsClassifier(),
    }
    print("\n5-fold CV, macro F1:")
    for name, m in models.items():
        s = cross_val_score(m, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"  {name:20s} {s.mean():.3f} ± {s.std():.3f}")
    print("\nTest set:")
    for name, m in models.items():
        m.fit(X_train, y_train)
        print(f"\n{name}\n", classification_report(y_test, m.predict(X_test)))


def tune_xgboost(X_train, y_train):
    grid = GridSearchCV(
        XGBClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
        cv=5, scoring="f1_macro",
    )
    grid.fit(X_train, y_train)
    print("\nBest params:", grid.best_params_)
    return grid.best_estimator_


# ----------------------------------------------------------------------- plots

def save_plots(df, model, X_test, y_test, y_pred, out):
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature importance (SHAP)")
    plt.tight_layout()
    plt.savefig(out / "shap_importance.png", dpi=150)
    plt.close()

    labels = ["Budget", "Heavy Data", "Premium", "Dissatisfied"]
    cm = confusion_matrix(y_test, y_pred, labels=list(PROFILE_NAMES))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    summary = df.groupby("Profile")[["Satisfaction_Score", "Q31val", "S7", "Q19"]].mean()
    summary.index = [PROFILE_NAMES[i] for i in summary.index]
    plt.figure(figsize=(7, 4.5))
    sns.heatmap(summary, annot=True, cmap="Blues")
    plt.title("Profile characteristics")
    plt.tight_layout()
    plt.savefig(out / "profile_summary.png", dpi=150)
    plt.close()

    if {"S1_1", "S1_2"} <= set(df.columns):
        df[["S1_1", "S1_2"]].sum().plot.pie(labels=["Male", "Female"], autopct="%1.1f%%")
        plt.title("Sex distribution")
        plt.ylabel("")
        plt.savefig(out / "sex_distribution.png", dpi=150)
        plt.close()
    if "S2" in df:
        ax = df["S2"].value_counts().sort_index().plot.bar()
        ax.set_xticklabels(["<10", "10-17", "18-25", "25-34", "35-44",
                            "45-54", "55-59", "60-70", ">71"][:len(ax.get_xticks())], rotation=45)
        plt.title("Age category distribution")
        plt.tight_layout()
        plt.savefig(out / "age_distribution.png", dpi=150)
        plt.close()


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="augmented survey (.xlsx)")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.data, sheet_name=args.sheet)
    print("Loaded:", raw.shape)
    df = preprocess(raw)

    features = select_features(df)
    X = df[features]
    elbow_plot(X, out)

    df["Profile"] = KMeans(n_clusters=N_PROFILES, random_state=SEED).fit_predict(X)
    print("\nProfile averages:\n",
          df.groupby("Profile")[["Satisfaction_Score", "Q31val", "S7", "Q19"]].mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, df["Profile"], test_size=0.2, random_state=SEED, stratify=df["Profile"])

    compare_models(X_train, y_train, X_test, y_test)
    best = tune_xgboost(X_train, y_train)
    y_pred = best.predict(X_test)
    print("\nFinal model:\n", classification_report(y_test, y_pred))

    # deployable pipeline: scaler + tuned XGBoost
    pipe = Pipeline([("scaler", StandardScaler()), ("model", best)]).fit(X_train, y_train)
    joblib.dump(pipe, out / "customer_profiling_model.pkl")

    save_plots(df, best, X_test, y_test, y_pred, out)

    sample = X_test.iloc[[0]]
    profile = PROFILE_NAMES[int(pipe.predict(sample)[0])]
    print(f"\nExample -> profile: {profile} | offer: {OFFERS[profile]}")
    print(f"Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
