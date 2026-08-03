# MIMIC-III ICU Mortality Prediction

Predicting in-hospital mortality (`HOSPITAL_EXPIRE_FLAG`) for ICU patients
using the MIMIC-III critical care database. Built in R using `tidymodels`,
evaluated via ROC-AUC through a private Kaggle competition.

## Overview

This project trains and compares several classification models — logistic
regression, decision trees, random forest, gradient boosting, and a neural
network — to predict the probability of in-hospital death based on a
patient's vitals, demographics, and diagnosis history recorded at ICU
admission.

Two approaches were explored:
1. **One-hot encoded diagnoses** — primary and secondary ICD9 diagnosis
   codes expanded into binary indicator columns.
2. **PCA-reduced diagnoses** — ICD9 codes grouped by 3-digit prefix (top 20
   most frequent + "Other"), one-hot encoded, then reduced to 5 principal
   components to control dimensionality.

## Data
Sourced from [MIMIC-III](https://mimic.physionet.org/), provided as a
preprocessed subset via Kaggle.

| File | Description |
|---|---|
| `mimic_train_X.csv` / `mimic_test_X.csv` | Vitals, demographics, and primary diagnosis per ICU stay |
| `mimic_train_y.csv` | Training labels — `HOSPITAL_EXPIRE_FLAG` |
| `MIMIC_diagnoses.csv` | Secondary/comorbidity ICD9 codes per patient |

*(Raw data not included in this repo — subject to MIMIC-III data use
terms.)*

## Feature Engineering

- Derived `entered_age` from `ADMITTIME` and `DOB`, capping ages above 89
  at 90 (matches MIMIC's own de-identification convention for very
  elderly patients).
- Merged primary (`ICD9_diagnosis`) and secondary (`MIMIC_diagnoses.csv`)
  diagnosis codes as binary indicator features.
- Aligned train/test columns so both sets have identical feature spaces
  (missing diagnosis indicators in test filled with 0).
- Converted `GENDER`, `ADMISSION_TYPE`, and `FIRST_CAREUNIT`
