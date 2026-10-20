# Capstone - Mobile Plan Recommendation

Customer profiling and personalized mobile plan recommendations from a telecom survey in Tunisia.

Capstone project, Manouba School of Engineering, 2024/2025
**Emna Saada, Souha Hamdeni**, mentored by Dr. Youssef Ben Halima (examiner: Dr. Ali Abbes).

## Overview

We had survey answers from 5,112 people about how they use mobile and internet services in Tunisia. The goal was to group users into profiles and recommend a plan that fits each one. The project has three parts:

- **Power BI dashboards** to explore usage, satisfaction and demographics
- **An ML pipeline** that sorts each respondent into one of 4 profiles and maps that profile to an offer
- **A web form** where a new user fills in the survey and gets a profile and an offer back

| Profile | Suggested offer |
|---|---|
| Budget-Conscious | Prepaid plan, 500 MB, 5 dt/month |
| Heavy Data | Postpaid, 10 GB, 20 dt/month, free social media |
| Premium | Unlimited data, priority support, 50 dt/month |
| Dissatisfied | Discounted plan, enhanced customer service, 15 dt/month |

## Pipeline

1. **Augmentation** (`src/augment_survey.py`): adds answers for the new questions Q40-Q46 (conditioned on income, age and education), then tops the data up to 10,000 respondents by sampling each answer from the real distribution. We tried SMOTE first, but it added noise to the categorical answers.
2. **Preprocessing**: mode/median imputation, one-hot for nominal answers, 5th-95th percentile clipping, and three engineered scores (`Satisfaction_Score`, `Device_Intensity`, `Internet_Usage_Score`). Columns with correlation above 0.8 are dropped.
3. **Feature selection**: a Random Forest trained on provisional K-Means labels keeps the top 20 features. The strongest were `Q20_11` and `Q29`.
4. **Profiles**: K-Means with k = 4, chosen with the elbow method.
5. **Classifier**: Logistic Regression, Random Forest, XGBoost and KNN are compared with 5-fold CV, then XGBoost is tuned with `GridSearchCV`. SHAP is used to explain the predictions.
6. **Export**: a `StandardScaler + XGBoost` pipeline is saved with joblib for the web form.

## Results

| Model | Macro F1 (5-fold CV) |
|---|---|
| Logistic Regression | 0.999 ± 0.001 |
| Random Forest | 0.999 ± 0.001 |
| XGBoost | 0.999 ± 0.001 |
| KNN | 0.997 ± 0.002 |

The tuned XGBoost reaches 1.00 on the test set, with only 4 errors out of 2,000 respondents.

**What the dashboards showed:**
- Peak internet use is between 18h and 22h.
- Ooredoo has the largest share of users (43.7%), followed by TTC (29.6%) and Orange (26.5%).
- 62% of users know their mobile voice prices, but only 26% know SMS/MMS prices and 18% know their call options.

## Limitations

- The profile labels come from K-Means, so the classifier is learning to reproduce the clusters. That is why the scores are close to perfect. They show the profiles can be predicted reliably from a short form, not that the profiles match real customer behaviour.
- About half of the 10,000 rows are synthetic, so the profiles should be re-checked on real data.
- The cluster ids to profile names (`PROFILE_NAMES`) were assigned by reading the cluster averages and need to be re-checked after any retraining.
- Many people did not want to answer the survey, so some groups (older or privacy-conscious users) are probably under-represented.

## Running it

```bash
pip install -r requirements.txt
python src/augment_survey.py data/clean03-05_needingadd.xlsx -o data/augmented_dataset.xlsx
python src/profiling_pipeline.py data/augmented_dataset.xlsx --out results
```

The survey data is not public, so it is not included in this repo. The figures and the trained model are written to `results/`.

The Power BI measures and label mappings are in `powerbi/measures.dax`.

## Repo layout

```
src/augment_survey.py       survey augmentation (Q40-Q46 + synthetic respondents)
src/profiling_pipeline.py   preprocessing, features, K-Means, models, SHAP, export
powerbi/measures.dax        DAX measures and code -> label columns
```
