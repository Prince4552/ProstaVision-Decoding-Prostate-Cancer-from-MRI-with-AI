# Radiomics Workflow

This folder contains the radiomics branch of the project. The goal is to classify clinically significant prostate cancer (`csPCa`) from multiparametric MRI (`T2W`, `ADC`, `DWI/HBV`) using radiomic features and classical machine-learning models.

The binary target label is:

- `0`: not clinically significant
- `1`: clinically significant (`ISUP >= 2`)

## Structure

```text
├── 1_extract_radiomics
│   ├── extract_radiomics.py
│   ├── Params_T2w.yaml
│   ├── Params_ADC.yaml
│   └── Params_DWI.yaml
└── 2_modeling
    ├── 0_build_concatenated_feature_table.py
    ├── 1_train_and_evaluate.py
    ├── 2_model_differences.py
    ├── 2a_gland_vs_full_differences.py
    └── 3_retrain_best_model_and_evaluate.py
```

## Step-by-Step Pipeline

### 1. Extract Radiomic Features for Each Modality

Script: [`1_extract_radiomics/extract_radiomics.py`](./1_extract_radiomics/extract_radiomics.py)

The script reads `artifacts/data.csv` and, for each study:

1. Loads the three sequences: `T2W`, `ADC`, and `DWI/HBV`.
2. Loads the prostate-gland mask.
3. Preprocesses each image:
   - Converts the image to `float32`
   - Corrects intensity inhomogeneity using `N4`
   - Reduces noise using anisotropic diffusion
4. Extracts radiomic features using two spatial approaches:
   - `gland`: only within the prostate gland
   - `full`: across the entire image using an all-ones mask
5. Saves a separate CSV for each modality and spatial approach.

Generated files in `artifacts/radiomics/`:

- `features_t2_gland.csv`
- `features_adc_gland.csv`
- `features_dwi_gland.csv`
- `features_t2_full.csv`
- `features_adc_full.csv`
- `features_dwi_full.csv`

### 2. Combine the Modalities into a Single Modeling Table

Script: [`2_modeling/0_build_concatenated_feature_table.py`](./2_modeling/0_build_concatenated_feature_table.py)

This script combines the previously generated CSV files into one final feature table for each spatial approach:

- `features_all_gland.csv`
- `features_all_full.csv`

Specifically, it:

1. Loads the `T2`, `ADC`, and `DWI` CSV files.
2. Removes PyRadiomics diagnostic columns (`diagnostics_*`).
3. Keeps `patient_id`, `study_id`, and `label`.
4. Adds modality-specific prefixes to avoid naming conflicts:
   - `t2_...`
   - `adc_...`
   - `dwi_...`
5. Keeps shape features from only one reference modality so they are not duplicated three times.
6. Creates a unique `sample_id` using `patient_id + "_" + study_id`.

## 3. Main Training with Repeated Cross-Validation

Script: [`2_modeling/1_train_and_evaluate.py`](./2_modeling/1_train_and_evaluate.py)

Six classifiers are compared:

- `SVM`
- `Logistic Regression`
- `Random Forest`
- `Naive Bayes`
- `KNN`
- `Gradient Boosting`

The default evaluation protocol is:

- `5 folds`
- `10 repetitions`
- Grouping by `patient_id`

This means that each classifier is evaluated across `50` validation folds, while ensuring that studies from the same patient should never be split between training and validation.

The script also creates the fold plan once and uses exactly the same folds for every model. This makes the comparison between classifiers fairer.

## 4. How Feature Selection Works

This is one of the most important parts of the pipeline and is also where confusion can easily arise.

### Main Idea

Feature selection is **not performed once on the entire dataset**. Instead, it is performed separately inside each fold using only that fold's training data.

This prevents data leakage. In other words:

- The validation data does not participate in feature selection.
- The validation data is used only at the end to measure performance.

### Exact Sequence Within Each Fold

When `--feature_strategy most_discriminant` is used, the code follows these steps:

1. Takes only the numerical radiomic features.
2. Removes metadata and columns that should not be used for modeling:
   - `patient_id`
   - `study_id`
   - `label`
   - `sample_id`
   - `mask_type`
   - `diagnostics_*` columns
3. Uses only the training set from the current fold.
4. Scores every feature individually:
   - Removes features with no variation or too many invalid values.
   - Attempts to assess normality.
   - If the distribution appears approximately normal, uses a `t-test`.
   - Otherwise, uses the `Mann-Whitney U` test.
   - Also calculates univariate `AUC` to rank features.
   - Calculates an optimal univariate threshold using Youden's index.
5. Corrects the `p-values` for multiple comparisons using Benjamini-Hochberg.
   - If features satisfy `q <= fdr_alpha`, they form the preferred feature pool.
   - If none survive, the script falls back to valid features ranked by univariate relevance.
6. Determines how many features can reasonably be retained for that fold.
   - The number is not a fixed value.
   - It depends on the size of the training set.
   - It also depends on the number of samples in the minority class.
   - The goal is to avoid selecting too many features relative to the amount of available data.
7. Removes redundant features based on correlation.
   - Candidate features are ordered by relevance.
   - The list is processed greedily.
   - If a feature is highly correlated with one that has already been selected, it is discarded.
   - By default, absolute Pearson correlation with a threshold of `0.90` is used.
8. Keeps the best remaining features until the allowed limit is reached.
9. Trains the classifier using this selected subset.
10. Evaluates the classifier on the validation partition of that fold.

### What This Means in Practice

- The selected feature set can change from one fold to another.
- This is not an error; it is what we expect when feature selection is performed correctly inside each training fold.
- Because the fold plan and feature-selection procedure are reused across all models, every classifier is evaluated under the same conditions.

### Feature-Selection Outputs

The following files are saved under `results/radiomics/<feature_strategy>/<mode>/feature_selection/`:

- `selected_features_by_fold.csv`: detailed feature-selection results for each fold
- `feature_selection_frequency.csv`: how often each feature is selected
- `top_selected_features.txt`: the most stable features for each classifier
- `recommended_features_by_classifier.txt`: a summarized feature list for each classifier

These outputs help answer two different questions:

- Which features did the model actually use in each fold?
- Which features were the most stable across folds?

## 5. What Happens After the `5 folds × 10 repetitions`

After completing the 50 folds for each classifier, the script continues with several additional steps.

### a. Save Fold-by-Fold Results

Training and validation metrics are exported for every fold:

- `AUC`
- `F1`
- `balanced accuracy`
- `MCC`
- `kappa`
- `sensitivity`
- `specificity`
- `PPV`
- `NPV`

The validation predictions and the list of features used in each fold are also saved.

### b. Build Out-of-Fold Predictions

First, the script creates a flat table with one row for each validation case. It then aggregates the repeated predictions.

This is important because, with `10` repetitions, the same case appears in the validation set multiple times. The script therefore:

1. Collects all predictions for that case.
2. Averages the predicted probability of the positive class for each case and classifier.
3. Applies the classification threshold, which is `0.5` by default.

The result is an aggregated **out-of-fold prediction for each case**.

### c. Summarize Performance at the Patient Level

Using these aggregated predictions, the script calculates more stable overall performance metrics and generates:

- `summary_metrics.csv`
- `auc_ci_summary.csv`
- Aggregated ROC curves
- Aggregated confusion matrices

It also performs patient-level stratified bootstrapping to obtain confidence intervals.

### d. Statistically Compare the Classifiers

When `--calculate_differences` is enabled, [`2_model_differences.py`](./2_modeling/2_model_differences.py) is run:

- Friedman test for an overall difference between models
- Paired Wilcoxon tests with Holm correction for pairwise comparisons

If you want to compare the `gland` and `full` approaches, you can use [`2a_gland_vs_full_differences.py`](./2_modeling/2a_gland_vs_full_differences.py).

## 6. What Happens with the Best Model

Script: [`2_modeling/3_retrain_best_model_and_evaluate.py`](./2_modeling/3_retrain_best_model_and_evaluate.py)

Once the best-performing classifier has been identified, the pipeline moves to a final stage that is closer to a definitive evaluation.

### Final Evaluation Workflow

1. Performs a final `80/20` group-based split using `GroupShuffleSplit`.
2. Repeats feature selection using only the training portion of this final split.
3. Restricts both the training and test sets to the selected features.
4. Optimizes hyperparameters using `BayesSearchCV` with grouped folds within the training set.
5. Saves the best trained estimator.
6. Evaluates the uncalibrated model on the held-out test set.
7. Calculates bootstrap confidence intervals on the test set.
8. Calibrates the predicted probabilities using Platt scaling (`sigmoid`).
9. Evaluates the calibrated model again.
10. Searches across classification thresholds and reports the threshold that maximizes `F1`.
11. Generates model explainability outputs using `SHAP` and `LIME` on both the training and test sets.

### How to Interpret This Stage

The repeated cross-validation is used to compare the models robustly. The final hold-out stage is then used to:

- Refine the best classifier
- Obtain a separate final evaluation
- Produce an interpretable and exportable model

## Main Outputs

Depending on the selected options, the following outputs are generated under `results/radiomics/<feature_strategy>/<mode>/`:

- Metrics for each fold
- Predictions for each fold
- Flat and aggregated out-of-fold predictions
- Feature selection results for each fold
- Aggregated performance summaries with confidence intervals
- ROC curves
- Statistical comparisons between models
- A folder containing the retrained best model, including:
  - `best_estimator.pkl`
  - `report.txt`
  - Calibration curves
  - Confusion matrices
  - `SHAP` and `LIME` explainability outputs

## Example Command

```bash
python train/radiomics/2_modeling/1_train_and_evaluate.py   --csv features_all_gland.csv   --data_pre artifacts/radiomics   --results_base results/radiomics   --feature_strategy most_discriminant   --n_splits 5   --n_repeats 10   --bootstrap_iterations 1000   --ci_level 0.95   --classification_threshold 0.5   --min_features 10   --max_features_cap 60   --samples_per_feature 25   --minority_samples_per_feature 8   --fdr_alpha 0.05   --correlation_threshold 0.90   --selection_n_jobs 8   --search_n_jobs 8   --search_iterations 50   --calculate_differences   --fine_tune_best_model
```

## Important Methodological Note

In the final hold-out script, the threshold search is performed directly on the test set. This can be useful for exploratory analysis, but if the goal is to report a strictly unbiased final evaluation, the threshold should be determined using the training data or an additional validation set rather than the final test set.
