# HiLabs Hackathon 2025: Patient Risk Identification
---
### **Solution by Team: Pika Pika**
Aayush Jaiswal, Manvi Bengani, Shubh Jain
## Introduction

This notebook presents our solution for the HiLabs Hackathon 2025. The challenge is to address a critical problem in Value-Based Care (VBC): **proactively identifying at-risk patients** to enable timely and targeted care interventions.

> **Problem Statement:** Predict Patient Risk levels from multi-source healthcare data to enable targeted, proactive care management.

The goal is to build a robust machine learning model that ingests multiple datasets—including patient demographics, diagnoses, visits, and care history—to generate a predictive risk score for each patient.
# Patient Risk Prediction — Feature Engineering & Modeling Pipeline

##  Overview

This repository builds a **comprehensive feature-engineering and predictive modeling pipeline** for patient risk and readmission prediction using multiple clinical data sources.  
Data from *patients, visits, diagnoses, and care records* are cleaned, merged, and transformed into a single analytical dataset for machine learning.

---

## Overall Approach & Data Architecture

### Data Sources

| Source File | Description | Key Columns |
|--------------|--------------|--------------|
| `patient.csv` | Base demographic and risk data | `patient_id`, `age`, `risk_score`, `hot_spotter_*` flags |
| `visit.csv` | Visit-level data | `patient_id`, `visit_type`, `visit_start_dt`, `visit_end_dt`, `prncpl_diag_nm`, `readmsn_ind` |
| `diagnosis.csv` | Chronic condition and history | `patient_id`, `condition_name`, `condition_description` |
| `care.csv` | Care and screening records | `patient_id`, `msrmnt_sub_type`, `last_care_dt`, `next_care_dt` |

---

##  Feature Selection Logic & Assumptions

###  1. **Visit Data (`visit.csv`)**

**Objective**: Capture utilization patterns, visit frequency, and recency.

| Feature | Logic / Method | Assumptions |
|----------|----------------|-------------|
| `diag_*` | Regex-based text binning of `prncpl_diag_nm` into 17 clinical categories (Respiratory, Musculoskeletal, etc.), which is then one-hot-encoded | Keywords cover majority of ICD descriptions |
| `visit_duration` | `visit_end_dt - visit_start_dt` | Missing end dates treated as `NaN` |
| `days_since_last_visit` | Grouped diff by patient on `visit_start_dt` | Assumes sorted chronological order |
| `early_readmit_flag` | Visits within 7 days of last discharge | Indicates potential readmission |
| `urgent_care_recent_flag` | URGENT CARE visit within 30 days of latest | Proxy for acute instability |
| `visit_type_*` | One-hot encoded - ER,Inpatient,Urgent| Distinct visit categories are exhaustive |
| Aggregates | `sum`, `mean`, `max` by patient | Represent cumulative and temporal patterns |

---

### 2. **Diagnosis Data (`diagnosis.csv`)**

**Objective**: Capture chronic disease prevalence and recency.

| Feature | Logic / Method | Assumptions |
|----------|----------------|-------------|
| `condition_status` | Keyword detection (“recent”, “past”) | Free-text parsing captures disease status |
| `has_DIABETES`, `has_HYPERTENSION`, `has_CANCER` | Boolean derived via one-hot encoding | Conditions captured by keywords |
| `condition_description` | Aggregated to “recent” if any recent record exists | Recency implies higher risk |

---

### 3. **Care Data (`care.csv`)**

**Objective**: Represent continuity of care and screening activity.

| Feature | Logic / Method | Assumptions |
|----------|----------------|-------------|
| `care_gap_days` | `(next_care_dt - last_care_dt)` bucketed to {0,1,2} | Missing handled via substitution |
| `care_gap_ind` | Converted to {0,1} | Indicates presence of screening |
| `msrmnt_sub_type` | Multi-label binarized for 8 key measurements (BP, HbA1c, etc.) | Encodes all possible types measurements |

---

### 4. **Patient & Risk Data (`patient.csv` + `risk.csv`)**

| Feature | Logic / Method | Assumptions |
|----------|----------------|-------------|
| `age`, `risk_score` | Base demographic/risk variables | No imputation needed |
| `hot_spotter_*_flag` | Converted from `{t,f}` → `{1,0}` | Consistent boolean normalization |
| `diagnosis_count` | Count of all nonzero `diag_*` per patient | Measures comorbidity load |

---

### Additional Cross Metrics
A comprehensive analysis of patient data reveals significant predictors of health risk, with chronic diseases such as cancer, diabetes, and hypertension emerging as top clinical drivers. The data underscores a strong correlation between higher risk scores and the presence of these conditions, the number of diagnoses, age, care gaps, and healthcare utilization patterns.

Patients with cancer have a substantially higher average risk score of approximately 4.46, a stark contrast to the ~1.58 average for those without the disease. Similarly, individuals with diabetes show an elevated average risk score of about 3.67, compared to ~1.43 for non-diabetics. Hypertension also plays a significant role, with affected patients having an average risk score of around 2.82, versus ~1.24 for those without.

The analysis further demonstrates that health risk escalates with the accumulation of unique diagnoses. An individual with no unique diagnoses has a baseline risk of about 1.12, which increases to ~2.31 with a single diagnosis, ~4.09 with two, and ~5.79 with three. A general flag for any chronic condition also signals a significantly higher risk, with an average score of ~6.56 compared to ~1.60 for those without such a flag. Thus the below features have been formed. 

Age is another critical factor, with risk scores progressively increasing across different age brackets:

0-20 years: ~0.67
21-40 years: ~1.08
41-60 years: ~1.91
61-80 years: ~2.80
81+ years: ~3.37
Moreover, indicators of care management, such as care gaps, are clear markers for higher-risk patients. Individuals with zero care gaps have an average risk of ~1.43, which dramatically jumps to ~7.38 with just one gap and ~8.68 for those with two to three gaps.

Finally, healthcare utilization patterns serve as strong indicators of risk, reflecting the severity or acute exacerbations of health issues. Frequent emergency room and inpatient visits are particularly telling. The risk rises from 1.4 for no ER visits to 7.0 for 6-20 visits. The increase is even more pronounced for inpatient visits, climbing from 1.52 for none to a steep 17.7 for 6-20 visits. Even a moderate increase in urgent care visits corresponds to a rise in risk. Overall, a higher total number of visits correlates with increased risk, with 6-20 total visits resulting in an average risk score of approximately 5.7.
| Feature | Description |
|----------|-------------|
| **total_visits** | Total number of visits (sum of ER, Inpatient, and Urgent Care) for each patient. |
| **er_ratio** | Fraction of Emergency Room visits = `ER / total_visits`. |
| **inpatient_ratio** | Fraction of Inpatient visits = `Inpatient / total_visits`. |
| **urgent_ratio** | Fraction of Urgent Care visits = `Urgent / total_visits`. |
| **visit_count** | Total number of visits recorded for a patient. |
| **total_readmissions** | Total number of readmissions across all visits. |

---

### Additional Cross Clinical Condition Metrics
| Feature | Description |
|----------|-------------|
| **has_DIABETES**, **has_HYPERTENSION**, **has_CANCER** | Binary indicators (1/0) representing presence of major chronic conditions. |
| **diagnosis_count** | Average number of diagnosis categories detected per patient. |
| **chronic_condition_count** | Total number of chronic conditions present (`DIABETES + HYPERTENSION + CANCER`). |
| **chronic_flag** | 1 if the patient has at least one chronic condition, else 0. |
| **chronic_complexity** | Weighted index capturing condition severity, based on stats descibed above:<br>`3.5×DIABETES + 2.8×HYPERTENSION + 4.5×CANCER`. |

---

## Model Architecture: Stacking Regressor with RidgeCV Meta-Learner

### Overview

This modeling stage implements a **Stacking Ensemble Regressor** that combines multiple base learners — each with distinct bias-variance characteristics — to predict the patient **risk score**.  
The ensemble leverages the complementary strengths of tree-based models (for non-linearity) and regularized linear models (for interpretability and generalization).

---

###  Model Structure
Multiple peer-reviewed studies have demonstrated that stacking ensemble methods outperform single-model or simpler ensemble approaches in medical risk prediction tasks (e.g., ICU mortality prediction, cancer prognosis, cardiovascular risk) — see for example Chen et al. (2024) for ICU mortality, Chakraborty et al. (2024) for stroke risk, and a comparative evaluation across 16 disease datasets showing stacking consistently beating boosting and bagging.

**Base Learners** :  Random Forest, XGBoost, LightGBM, RidgeCV 
- Capture diverse signal patterns in the feature space 

**Meta-Learner** : RidgeCV 
- Aggregates base learner predictions using L2-regularized regression 

**Cross-Validation** : 5-fold KFold 
- Reduces variance and ensures robust out-of-fold training for stacking 

**Scaler** : StandardScaler 
- Ensures numeric features are standardized for consistent model behavior 


##  Regression Metrics

| Metric | Training Set | Testing Set |
|---------|---------------|--------------|
| **R-squared (R²)** | 0.67 | 0.52 |
| **Mean Absolute Error (MAE)** | 0.73 | 0.80 |
| **Root Mean Squared Error (RMSE)** | 1.49 | 1.48 |

# Project Setup

Follow the steps below to set up and run the project:

1. **Download all the project files.**

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make the setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

4. **Run the setup script from the terminal:**
   ```bash
   ./setup.sh
   ```