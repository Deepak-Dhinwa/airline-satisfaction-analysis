# Airline Satisfaction Analysis
### Data-Centric ML Pipeline: Feature Engineering, Multicollinearity Handling & Ensemble Modeling

---

## Resume Summary

- Performed in-depth EDA and missing value pattern analysis (MCAR/MAR/MNAR) on 103,904 records, applying regression-based imputation for arrival delay using departure delay as predictor (R²=0.93) rather than naive mean/median fill
- Engineered features using domain-driven interaction terms (Travel Type × Class), entropy-aware ordinal zero handling, and binary engagement flags; resolved multicollinearity via VIF analysis (18 features above VIF=5 reduced to zero) and applied PCA to survey rating block — recovering interpretable latent dimensions (Cabin Comfort, Digital Experience, Staff Service)
- Compared filter, wrapper, and embedded feature selection methods (Mutual Information, RFE with Random Forest, L1 Logistic Regression) to identify optimal 23-feature subset from 28 engineered features before modeling
- Built and evaluated Logistic Regression, Random Forest, and XGBoost classifiers using stratified 5-fold CV; XGBoost achieved ROC-AUC=0.987, F1=0.935 on held-out test set; interpreted predictions using SHAP values to derive actionable passenger experience insights

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Pipeline Overview](#project-pipeline-overview)
3. [Phase 1 — Structural Audit](#phase-1--structural-audit)
4. [Phase 2 — Missing Value Analysis](#phase-2--missing-value-analysis)
5. [Phase 3 — Ordinal Feature Handling](#phase-3--ordinal-feature-handling)
6. [Phase 4 — Distribution Analysis & Transformation](#phase-4--distribution-analysis--transformation)
7. [Phase 5 — Categorical Encoding & Feature Engineering](#phase-5--categorical-encoding--feature-engineering)
8. [Phase 6 — Outlier Detection](#phase-6--outlier-detection)
9. [Phase 7 — Scaling Strategy](#phase-7--scaling-strategy)
10. [Phase 8 — Multicollinearity Analysis (VIF)](#phase-8--multicollinearity-analysis-vif)
11. [Phase 9 — PCA on Service Rating Block](#phase-9--pca-on-service-rating-block)
12. [Phase 10 — Feature Selection](#phase-10--feature-selection)
13. [Phase 11 — Modeling & Evaluation](#phase-11--modeling--evaluation)
14. [Phase 12 — SHAP Interpretation](#phase-12--shap-interpretation)
15. [Key Business Insights](#key-business-insights)
16. [Tech Stack](#tech-stack)

---

## Dataset

**Source:** [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

| Property | Value |
|---|---|
| Train rows | 103,904 |
| Test rows | 25,976 |
| Features | 23 (after engineering) |
| Target | `satisfaction` — binary (satisfied / neutral or dissatisfied) |
| Class balance | 43.3% satisfied / 56.7% dissatisfied |

**Feature types in raw data:**
- 4 categorical features (Gender, Customer Type, Type of Travel, Class)
- 14 ordinal rating features (1–5 satisfaction surveys)
- 3 numerical features (Age, Flight Distance, Departure/Arrival Delay)
- 1 binary target

---

## Project Pipeline Overview

```
Raw Data
    │
    ├── Structural Audit (junk columns, dtype inspection, duplicates)
    │
    ├── Missing Value Analysis (MCAR / MAR / MNAR classification)
    │       └── Regression imputation for Arrival Delay (R²=0.93)
    │
    ├── Ordinal Zero Handling (3-tier strategy)
    │       ├── Tier 1 (high zero %): preserve + binary "not used" flag
    │       └── Tier 2/3 (low zero %): replace with median (data entry errors)
    │
    ├── Distribution Analysis (skewness, kurtosis, modality)
    │       ├── Yeo-Johnson transform → Flight Distance (skew: 1.1 → 0.03)
    │       └── Log1p + binary flag → Delay columns (zero-inflated structure)
    │
    ├── Categorical Encoding & Feature Engineering
    │       ├── Binary encode: Gender, Customer Type, Type of Travel
    │       ├── One-hot encode: Class (no false ordinality)
    │       └── Interaction features: Travel Type × Class (3 combinations)
    │
    ├── Outlier Detection (Z-score + Modified IQR + Isolation Forest)
    │       └── Winsorize delay columns at 99th percentile (cap, not drop)
    │
    ├── Scaling (per-feature, not global)
    │       ├── StandardScaler → Age, Flight Distance Transformed
    │       ├── RobustScaler → Delay columns (skewed, outliers remain)
    │       └── No scaling → Ordinal ratings, binary features
    │
    ├── Multicollinearity Analysis (VIF)
    │       ├── 18 features with VIF > 5 (max: 26.3)
    │       ├── Drop redundant raw delay columns
    │       └── PCA on 14 service rating columns → 9 orthogonal components
    │
    ├── Feature Selection (3 methods compared)
    │       ├── Filter: Mutual Information
    │       ├── Wrapper: RFE with Random Forest
    │       └── Embedded: L1 Logistic Regression
    │
    └── Modeling (Stratified 5-Fold CV)
            ├── Logistic Regression — AUC: 0.964
            ├── Random Forest      — AUC: 0.983
            └── XGBoost            — AUC: 0.987 ✓ (best)
                    └── SHAP interpretation
```

---

## Phase 1 — Structural Audit

**What we did:** Before any transformation, audited the raw data for structural issues that silent later errors.

**Findings:**

| Issue | Finding | Action |
|---|---|---|
| Junk columns | `Unnamed: 0` and `id` — index artifacts | Dropped immediately |
| Float dtype signal | `Arrival Delay in Minutes` is float64 while all others are int64 | Flagged as containing NaN values |
| Ordinal as int64 | 14 rating columns stored as integers | Pandas treats them as continuous — requires careful handling |
| Target as object | `satisfaction` is string | Encode only after EDA |
| Duplicates | 0 duplicate rows | No action needed |

**Key principle:** `Arrival Delay` being `float64` while `Departure Delay` is `int64` is the *only* signal you need to detect missing values before even calling `.isnull()`. Dtype inference tells you something about the data's history.

---

## Phase 2 — Missing Value Analysis

**What we did:** Classified the missingness mechanism before choosing an imputation strategy.

**The MCAR / MAR / MNAR Framework:**

| Type | Meaning | Safe to impute? |
|---|---|---|
| MCAR — Missing Completely At Random | No relationship between missingness and any variable | Yes, freely |
| MAR — Missing At Random | Missingness depends on *other observed variables* | Yes, conditionally |
| MNAR — Missing Not At Random | Missingness depends on the *missing value itself* | Dangerous — bias risk |

**Finding:** `Arrival Delay in Minutes` has 310 missing values (0.298%). Testing the missingness mechanism:

| Group | Mean Departure Delay | Count |
|---|---|---|
| Arrival delay present | 14.7 min | 103,594 |
| Arrival delay missing | **37.4 min** | 310 |

Missingness is strongly related to departure delay — flights that departed later were more likely to have missing arrival data (operationally chaotic flights with data entry gaps). This is **MAR**, not MCAR.

**Decision:** Regression imputation using Departure Delay as predictor.

```
Arrival Delay = 0.9802 × Departure Delay + 0.7224
R² = 0.9322
```

The coefficient of 0.98 is physically sensible: airlines recover ~2% of delay in-air on average. The imputation is domain-justified, not just statistically convenient.

**Critical rule:** The regression was fit on training data only, then applied to test data using the same fitted model. Re-fitting on test data would constitute data leakage.

---

## Phase 3 — Ordinal Feature Handling

**What we did:** Identified and handled the zero values in ordinal rating columns (expected range: 1–5, actual range: 0–5).

**Finding:** All 14 rating columns contain zeros. A zero in a 1–5 satisfaction rating is not "very dissatisfied" — it means the passenger **did not use or rate that service**. Treating 0 as a valid rating distorts means, correlations, and model learning.

**Three-tier strategy based on zero frequency:**

| Tier | Zero % | Interpretation | Action |
|---|---|---|---|
| Tier 1 (high) | 2–5% | Structural non-usage — optional/digital services | Preserve zero + create binary "not used" flag |
| Tier 2 (low) | 0.01–0.5% | Data entry error — unavoidable services | Replace with column median |
| Tier 3 (near-zero) | < 0.01% | Noise | Replace with column median |

**Tier 1 columns (engagement flags created):**

| Feature | Zero % | Insight |
|---|---|---|
| Departure/Arrival time convenient | 5.1% | Passengers on corporate bookings don't rate timing |
| Ease of Online booking | 4.3% | Some passengers use travel agents |
| Inflight wifi service | 3.0% | Significant segment doesn't use inflight wifi |
| Online boarding | 2.3% | Subset uses traditional boarding pass |

**Validation:** Zero-raters on `Departure/Arrival time convenient` show 47.5% satisfaction vs 43.1% for non-zero-raters — confirming zeros represent a distinct behavioural segment, not a low-satisfaction rating.

**Baggage handling** has zero zeros — the only column. Every passenger experiences baggage handling, so no one skipped rating it. Domain logic confirms the data.

---

## Phase 4 — Distribution Analysis & Transformation

**What we did:** Measured skewness and kurtosis for all numerical features and applied appropriate transformations.

**Diagnostic results:**

| Feature | Skewness | Kurtosis | Classification |
|---|---|---|---|
| Age | -0.005 | -0.720 | Near-perfect symmetric |
| Flight Distance | 1.109 | 0.268 | Moderately right-skewed |
| Departure Delay | 6.734 | 100.262 | Zero-inflated, extreme outliers |
| Arrival Delay | 6.565 | 93.503 | Zero-inflated, extreme outliers |

**Transformation decisions:**

**Age:** No transformation. Skewness ≈ 0. Ready to use as-is.

**Flight Distance:** Yeo-Johnson transform. Handles zeros, finds optimal λ to maximize normality. Result: skewness 1.109 → -0.027.

**Delay columns:** Standard log transform fails here. Median = 0 means >50% of flights have zero delay. These are **zero-inflated distributions** — a mixture of two processes:
1. Flights departing on time (zero mass)
2. Delayed flights (right-skewed continuous distribution)

Strategy: `log1p` transform (preserves the zero mass: log(0+1) = 0) + binary flag (`delayed = delay > 0`). This preserves the two-process structure instead of collapsing it.

**Results after transformation:**

| Feature | Before Skewness | After Skewness | Before Kurtosis | After Kurtosis |
|---|---|---|---|---|
| Flight Distance | 1.109 | -0.027 | 0.268 | -0.840 |
| Departure Delay | 6.734 | 0.917 | 100.262 | -0.557 |
| Arrival Delay | 6.565 | 0.871 | 93.503 | -0.641 |

Kurtosis drop from ~100 to ~-0.6 for delay columns is the key result — extreme tail behaviour is eliminated while the zero mass structure is preserved through the binary flags.

**Domain insight from binary flags:** 43.5% of flights experience departure delay. 44.0% experience arrival delay. The 0.5% difference represents flights that arrived late despite on-time departure — air traffic control holds, headwinds, etc. The binary features capture this distinction cleanly.

---

## Phase 5 — Categorical Encoding & Feature Engineering

**What we did:** Analyzed categorical features before encoding, then chose encoding strategy based on data properties.

**Satisfaction rates by category:**

| Feature | Category | Satisfied % | Signal Strength |
|---|---|---|---|
| Gender | Female / Male | 42.7% / 43.9% | **Weak** (1.2pp gap) |
| Customer Type | Loyal / Disloyal | 47.7% / 23.7% | Moderate (24pp gap) |
| Type of Travel | Business / Personal | 58.3% / 10.2% | **Strong** (48pp gap) |
| Class | Business / Eco / Eco Plus | 69.4% / 18.6% / 24.6% | **Strong** (51pp gap) |

**Encoding decisions:**

| Feature | Strategy | Reason |
|---|---|---|
| Gender | Binary (Male=1) | 2 values, no ordinality |
| Customer Type | Binary (Loyal=1) | 2 values, no ordinality |
| Type of Travel | Binary (Business=1) | 2 values, no ordinality |
| Class | One-hot (3 dummies) | 3 values, **no natural ordinality** — Eco Plus satisfaction (24.6%) is not between Business (69.4%) and Eco (18.6%) |

**Why not label encode Class?** Label encoding assigns 0/1/2 to categories, implying equal spacing and a meaningful order. Eco Plus is priced between Business and Eco but satisfaction doesn't follow price — it underperforms. Label encoding would lie to the model about the relationship.

**Interaction Feature Engineering:**

The Travel Type × Class crosstab revealed strong non-linear joint effects:

| Segment | N (% of data) | Satisfied % |
|---|---|---|
| Business Travel + Business Class | ~33% | 72.0% |
| Business Travel + Eco | ~19% | 29.6% |
| Business Travel + Eco Plus | ~4% | 39.3% |
| Personal Travel + Business Class | ~2% | **12.2%** |
| Personal Travel + Eco | ~25% | 10.2% |
| Personal Travel + Eco Plus | ~3% | **8.7%** |

Three domain-driven interaction features created:
1. `business_travel_x_business_class` — the high-satisfaction anchor segment
2. `personal_travel_x_eco` — the high-volume dissatisfied segment
3. `ecoplus_personal` — the product-failure segment (paid premium, least satisfied)

**Business insight:** 99.14% of disloyal customers are business travelers. Customer churn is almost entirely a corporate account problem, not a leisure traveler problem. This reframes the airline's retention strategy entirely.

---

## Phase 6 — Outlier Detection

**What we did:** Applied three detection methods simultaneously to distinguish real extreme values from data errors.

**Why IQR alone is wrong for this data:**

The standard 1.5×IQR fence on departure delay produces a ceiling of 48 minutes. Under this rule, 9,318 rows (8.97% of data) are flagged as outliers. These are not errors — they are real delayed flights. The IQR method fails because Q1=0 and Q3=12 (zero-inflated data), making IQR=12 and the fence absurdly narrow.

**Three methods applied:**

| Method | Departure Delay Outliers | Interpretation |
|---|---|---|
| Z-score (\|z\| > 3) | 2,222 (2.1%) | Statistically extreme values |
| Modified IQR (3× fence) | 9,318 (8.97%) | Fails on zero-inflated data |
| Isolation Forest (multivariate) | 2,079 (2.0%) | Genuine anomalies in feature space |

**Isolation Forest anomaly profile:**

| Group | Mean Departure Delay | Mean Arrival Delay | Satisfied % |
|---|---|---|---|
| Normal rows | 10.9 min | 11.3 min | 43.4% |
| Anomalous rows | **204.8 min** | **208.8 min** | 38.6% |

Anomalous rows are real severely delayed flights — lower satisfaction (38.6% vs 43.4%) confirming they carry signal, not noise.

**Decision: Winsorize at 99th percentile, do not drop.**

Dropping 2,000+ rows removes real signal. Winsorizing caps values at the 99th percentile (182 min departure, 184 min arrival) while preserving all rows. This treats all extreme delays as "equally extreme" — a distinction between 400 and 1,592 minutes of delay is unlikely to matter for satisfaction prediction.

---

## Phase 7 — Scaling Strategy

**What we did:** Applied different scalers to different features based on their distributional properties.

**Why global StandardScaler is wrong:**

StandardScaler computes z = (x - μ) / σ. For skewed features with remaining outliers, μ and σ are distorted — a single extreme value inflates σ, compressing all other values toward zero.

**Per-feature scaling decisions:**

| Feature Group | Scaler | Reason |
|---|---|---|
| Age, Flight Distance Transformed | StandardScaler | Near-normal after transformation |
| All delay columns | RobustScaler | Skewed, zero-inflated, outliers remain — uses median/IQR instead of mean/std |
| Ordinal ratings (1–5) | None | Already bounded, tree models don't require scaling |
| Binary/dummy features | None | Scaling binary features destroys their interpretability |

**RobustScaler formula:** x' = (x - median) / IQR

Resistant to outliers because median and IQR are percentile-based statistics unaffected by extreme values.

**Critical rule:** Scalers fitted on training data only. Test data transformed using the same fitted scaler. Fitting on full dataset leaks test distribution into training — a form of data leakage.

---

## Phase 8 — Multicollinearity Analysis (VIF)

**What we did:** Computed Variance Inflation Factor for all continuous features to detect and resolve multicollinearity.

**VIF formula:** VIF_j = 1 / (1 - R²_j), where R²_j is the R² from regressing feature j on all other features.

**Interpretation:** VIF=1 (no correlation), VIF=5 (moderate), VIF=10 (severe — standard removal threshold), VIF=∞ (perfect collinearity).

**Initial VIF results — 18 of 20 features above threshold:**

| Feature | VIF | Issue |
|---|---|---|
| Inflight entertainment | 26.3 | Halo effect in survey ratings |
| Departure Delay (raw) | 25.0 | Redundant with transformed version |
| Arrival Delay (raw) | 24.6 | Redundant with transformed version |
| Inflight service | 20.9 | Halo effect |
| Cleanliness | 20.5 | Halo effect |
| ... (13 more above 5) | ... | |
| Age | 1.09 | ✓ Clean |
| Flight Distance Transformed | 1.06 | ✓ Clean |

**Halo effect confirmed:** Passengers who rate one service highly tend to rate all services highly — not because all services are equally good, but because overall satisfaction colors individual ratings. This is a well-documented survey bias.

**Resolution strategy:**
1. Drop raw delay columns — keep log-transformed versions (better distributional properties, same information)
2. Apply PCA to the 14 service rating columns — see Phase 9

**Final VIF after resolution:** All features below 4.5. Zero features above the VIF=5 threshold.

---

## Phase 9 — PCA on Service Rating Block

**What we did:** Applied PCA to the 14 service rating columns as a principled response to halo-effect-driven multicollinearity.

**Why PCA here specifically:** PCA finds orthogonal directions of maximum variance. Principal components are mathematically guaranteed to have zero correlation with each other — VIF=1 by construction. It doesn't arbitrarily drop features; it finds the actual latent structure in the data.

**Explained variance by component:**

| Component | Variance % | Cumulative % |
|---|---|---|
| PC1 | 26.89% | 26.89% |
| PC2 | 18.49% | 45.38% |
| PC3 | 14.11% | 59.48% |
| PC4 | 8.18% | 67.67% |
| PC5 | 6.62% | 74.29% |
| PC6–PC9 | 15.77% | 90.06% |

**Decision:** Retain 9 components (90% variance threshold). The slow accumulation (9 components to reach 90%) reveals that service ratings are not as redundant as VIF suggested — passengers do meaningfully differentiate between service types, but the halo effect creates statistical correlation that inflates VIF.

**Component interpretation from loadings:**

| Component | Top Features | Interpretation |
|---|---|---|
| PC1 (26.9%) | Inflight entertainment, Cleanliness, Seat comfort, Food | **Cabin Physical Comfort** |
| PC2 (18.5%) | Ease of Online booking, Time convenient, Inflight wifi | **Digital & Pre-Flight Experience** |
| PC3 (14.1%) | On-board service, Inflight service, Baggage handling | **Human Staff Service Quality** |
| PC4 (8.2%) | Online boarding vs Time convenient (contrast) | **Digital vs Timing Trade-off** |
| PC5 (6.6%) | Checkin service (dominant, 0.811 loading) | **Check-in Experience (isolated)** |

PCA recovered the three natural service pillars — Cabin, Digital, Staff — without being told they exist. These align precisely with what airline strategists define as their core service dimensions.

---

## Phase 10 — Feature Selection

**What we did:** Applied three philosophically different feature selection methods and compared their agreement.

**Three methods:**

| Method | Type | What it measures |
|---|---|---|
| Mutual Information | Filter | Statistical dependency between each feature and target (marginal, model-free) |
| RFE with Random Forest | Wrapper | Feature importance accounting for interactions (computationally expensive) |
| L1 Logistic Regression | Embedded | Features whose coefficients survive L1 regularization penalty |

**Key agreements (selected by all three):**
- `service_PC1` (Cabin Comfort) — top MI, RFE selected, L1=0.898
- `is_business_travel` — strong signal everywhere, L1=3.043
- `is_loyal` — consistent across methods, L1=2.684
- `business_travel_x_business_class` — 2nd MI, RFE selected, L1=0.897
- All service PCs (PC1–PC9) — RFE selected all

**Notable disagreement — `Inflight_wifi_service_not_used`:**
- MI score: 0.026 (near bottom — 20th of 28)
- L1 coefficient: **11.032** (largest in entire model)
- RFE: selected

This apparent contradiction reveals a fundamental difference between filter and embedded methods. MI measures *marginal* dependency — this feature alone doesn't strongly predict satisfaction. L1 measures *conditional* importance — given all other features simultaneously, this flag becomes a sharp discriminator. A business traveler who didn't engage with wifi is a very different profile than a personal traveler who didn't. Filter methods miss features whose importance is conditional on interactions with other features.

**Confirmed noise — `is_male` (Gender):**
- MI: 0.007 (27th of 28)
- RFE rank: 6 (not selected)
- L1 coefficient: 0.021 (near-zero, not zeroed)

Every method agrees: gender has no predictive value. Consistent with the 1.2 percentage point satisfaction gap observed in EDA.

**Features dropped:** `is_male`, both binary delay flags, `Departure_Delay_in_Minutes_transformed`, `Online_boarding_not_used`

**Final feature set:** 23 features retained from 28 engineered features.

---

## Phase 11 — Modeling & Evaluation

**Cross-validation strategy:** Stratified 5-fold CV — each fold maintains the same 43.3%/56.7% class ratio as the full dataset. Primary metric: ROC-AUC (unaffected by class imbalance, measures discriminative ability across all thresholds).

**Train/test AUC gap** is the overfitting diagnostic. Gap > 0.05 indicates memorization.

### Cross-Validation Results

| Model | CV ROC-AUC | CV F1 | CV Precision | CV Recall | Train AUC | Overfit Gap |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9643 ± 0.0011 | 0.8864 ± 0.0036 | 0.8873 ± 0.0044 | 0.8856 ± 0.0048 | 0.9644 | **0.0001** |
| Random Forest | 0.9833 ± 0.0005 | 0.9240 ± 0.0019 | 0.9435 ± 0.0025 | 0.9053 ± 0.0030 | 0.9942 | 0.0109 |
| XGBoost | **0.9870 ± 0.0004** | **0.9330 ± 0.0022** | **0.9456 ± 0.0014** | **0.9208 ± 0.0035** | 0.9932 | **0.0062** |

### Held-Out Test Set Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9642 | 0.8862 | 0.8855 | 0.8869 |
| Random Forest | 0.9833 | 0.9256 | 0.9425 | 0.9094 |
| **XGBoost** | **0.9868** | **0.9346** | **0.9456** | **0.9239** |

### Model Analysis

**Logistic Regression** achieves ROC-AUC of 0.964 — remarkably strong for a linear model. The near-zero overfitting gap (0.0001) demonstrates that our feature engineering encoded non-linear relationships explicitly (interaction features), allowing a linear model to capture them. This validates the data-centric approach: good features reduce the need for complex models.

**Random Forest** shows the largest overfitting gap (0.0109) — the forest memorizes some training patterns. Tightenable with stronger regularization (`min_samples_leaf`, `max_depth`), but performance is still excellent.

**XGBoost** is the best model: highest AUC, F1, and Recall; half the overfitting of Random Forest; tightest cross-validation variance (±0.0004). Sequential boosting with subsampling provides natural regularization.

---

## Phase 12 — SHAP Interpretation

**What we did:** Applied SHAP (SHapley Additive exPlanations) to the final XGBoost model to convert predictions into interpretable feature contributions.

**SHAP Shapley value formula:**

φⱼ = Σ [|S|!(p-|S|-1)!/p!] × [f(S∪{j}) - f(S)]

Where S is every subset of features not containing j, p is total features, and f(S) is model output using only features in S. For each prediction, SHAP tells you exactly how much each feature pushed the prediction toward satisfied or dissatisfied.

### Global Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | service_PC1 | 1.742 | Cabin comfort is the dominant driver |
| 2 | is_business_travel | 1.059 | Travel purpose — 2nd strongest |
| 3 | is_loyal | 0.707 | Customer loyalty status |
| 4 | service_PC4 | 0.671 | Digital vs timing trade-off |
| 5 | service_PC2 | 0.583 | Digital/pre-flight experience |
| 6 | business_travel_x_business_class | 0.515 | Interaction term — validates engineering |
| 7 | Inflight_wifi_service_not_used | 0.321 | Wifi engagement segment |
| 8 | service_PC3 | 0.298 | Staff service quality |
| 9 | Arrival delay (transformed) | 0.267 | Delay impact |
| 10 | Age | 0.226 | Passenger age profile |

### Single Prediction Breakdown

**Satisfied passenger (predicted probability: 0.979):**

| Feature | Value | SHAP | Direction |
|---|---|---|---|
| service_PC1 | 3.289 | +3.092 | High cabin comfort → strongly satisfied |
| is_business_travel | 1.0 | +1.095 | Business traveler → satisfied |
| is_loyal | 1.0 | +0.717 | Loyal customer → satisfied |
| Arrival delay | 1.442 | -0.495 | Some delay → pulls toward dissatisfied |
| business_travel_x_business_class | 0.0 | -0.597 | Not in premium segment → mild negative |

**Dissatisfied passenger (predicted probability: 0.006):**

| Feature | Value | SHAP | Direction |
|---|---|---|---|
| service_PC1 | -4.070 | -2.867 | Very low cabin comfort → strongly dissatisfied |
| is_loyal | 0.0 | -1.231 | Disloyal customer → dissatisfied |
| service_PC2 | -0.604 | -0.689 | Poor digital experience → dissatisfied |
| is_business_travel | 1.0 | +0.383 | Business traveler → partially offsets |

---

## Key Business Insights

**1. Cabin comfort is the single most important satisfaction driver.**
service_PC1 (Inflight entertainment, Cleanliness, Seat comfort, Food) has the highest SHAP importance (1.742) — more than 1.6× the next feature. Airlines seeking to improve satisfaction should prioritize physical cabin investment over digital services.

**2. Travel purpose determines satisfaction ceiling more than any service quality metric.**
Personal travelers have a 10.2% satisfaction rate regardless of class. Even personal travelers in Business class are only 12.2% satisfied. No amount of service improvement can overcome the expectation gap for leisure passengers on a product designed for business travel.

**3. Eco Plus is a failing product.**
Personal Travel + Eco Plus passengers (8.7% satisfied) are *less* satisfied than Personal Travel + Eco passengers (10.2% satisfied). Passengers paying more are less satisfied — a clear product-market fit failure.

**4. Customer churn is a corporate account problem.**
99.14% of disloyal customers are business travelers. Retention efforts should focus entirely on corporate accounts, not leisure passengers.

**5. Wifi engagement is a stronger satisfaction signal than the rating itself.**
The binary flag "did not use inflight wifi" received L1 coefficient of 11.032 — the largest in the model — despite low mutual information with the target. Whether a passenger engages with wifi is more predictive than how they rate it.

**6. A linear model with good features nearly matches ensemble performance.**
Logistic Regression achieves ROC-AUC=0.964 vs XGBoost's 0.987. The 2.3 point gap required a gradient boosted ensemble to close. This demonstrates that feature engineering — not model complexity — drives the majority of predictive performance.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and EDA |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scipy.stats` | Skewness, kurtosis, distribution testing |
| `sklearn.preprocessing` | StandardScaler, RobustScaler, PowerTransformer |
| `sklearn.decomposition` | PCA |
| `sklearn.ensemble` | RandomForestClassifier, IsolationForest |
| `sklearn.linear_model` | LogisticRegression (L1) |
| `sklearn.feature_selection` | mutual_info_classif, RFE |
| `sklearn.model_selection` | StratifiedKFold, cross_validate |
| `statsmodels` | variance_inflation_factor |
| `xgboost` | XGBClassifier |
| `shap` | TreeExplainer, SHAP values |

---

*Dataset: Airline Passenger Satisfaction — Kaggle*
