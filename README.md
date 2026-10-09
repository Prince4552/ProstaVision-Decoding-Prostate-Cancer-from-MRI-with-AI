# Prostate Cancer Classification from MRI Using AI

This repository contains the code and results for a machine-learning project that uses multi-parametric MRI (mpMRI) to classify clinically significant prostate cancer (`csPCa`).

The project explores two main approaches:
- **Radiomics + classical machine learning** — extracting measurable image features and using traditional ML classifiers.
- **Deep learning** — using deep learning models to learn useful patterns directly from MRI data.

The project also focuses on making the evaluation reliable and reproducible. In particular, patients are kept separated between training and validation, feature selection is performed without using validation data, and model predictions are analysed in detail.

## What Is the Project Trying to Predict?

The main goal is to determine whether a case contains **clinically significant prostate cancer (`csPCa`)**.

In this project, `csPCa` is defined as **ISUP grade group `>= 2`**.

The MRI data uses three axial sequences:

- **T2-weighted (`T2W`)**
- **Apparent diffusion coefficient (`ADC`)**
- **High b-value diffusion-weighted imaging (`DWI` / `HBV`)**

## Repository Structure

```text
├── artifacts/
│   ├── data.csv                     # Cohort table with image paths, labels, and metadata
│   └── radiomics/                  # Extracted modality-specific radiomics CSV files
├── data_analysis/                  # Exploratory notebooks and descriptive analyses
├── data_structuring/               # Notebook used to assemble the cohort CSV
├── results/                        # Model outputs, comparisons, hold-out evaluation, plots
├── train/
│   ├── common/                     # Shared utilities for reproducibility and radiomics helpers
│   ├── compare_approaches/         # Radiomics vs deep learning comparison scripts
│   ├── deep_learning/
│   └── radiomics/
├── z_figures/
└── z_report/
```

# How the Project Works

The complete workflow can be understood as a sequence of steps:

**MRI data → radiomics features → feature selection → machine-learning models → repeated evaluation → model comparison → final model optimization and interpretation**

## 1. Build the Cohort Table

The project starts with:

`artifacts/data.csv`

This file brings together the information needed for each case, including:

- patient and study identifiers
- binary label (`case_csPCa`)
- paths to the three MRI sequences
- whole-gland segmentation path
- additional clinical and image metadata

This table is created using the notebooks in:

`data_structuring/`

## 2. Extract Radiomics Features from the MRI

Script:

`train/radiomics/1_extract_radiomics/extract_radiomics.py`

For every case and each MRI modality (`T2W`, `ADC`, and `DWI`), the script performs the following steps:

1. Loads the MRI volume.
2. Applies preprocessing:
   - converts the image to `float32`
   - performs N4 bias-field correction
   - performs curvature anisotropic diffusion denoising
3. Uses the whole-gland mask for the gland-focused analysis.
4. Creates an all-ones mask for the full-volume analysis.
5. Runs PyRadiomics using the configuration specified in the modality-specific YAML file.

This produces six CSV files inside `artifacts/radiomics/`:

- `features_t2_gland.csv`
- `features_adc_gland.csv`
- `features_dwi_gland.csv`
- `features_t2_full.csv`
- `features_adc_full.csv`
- `features_dwi_full.csv`

In simple terms, this step converts the MRI images into a large set of numerical measurements called **radiomics features**.

## 3. Combine the Radiomics Features

Script:

`train/radiomics/2_modeling/0_build_concatenated_feature_table.py`

The six modality-specific CSV files are combined into one table for each spatial setting:

- `features_all_gland.csv`
- `features_all_full.csv`

Important details:

- rows are matched using `patient_id`, `study_id`, and `label`
- feature names are prefixed by modality (`t2_`, `adc_`, `dwi_`)
- shape features are retained from only one reference modality to avoid redundant duplicates
- a unique `sample_id = patient_id + "_" + study_id` is created

The result is a single table containing the radiomics information that can be given to the machine-learning models.

## 4. Train and Compare Six Classical ML Models

Script:

`train/radiomics/2_modeling/1_train_and_evaluate.py`

This is the main radiomics benchmarking script.

It evaluates six classical machine-learning classifiers:

- SVM
- Logistic Regression
- Random Forest
- Naive Bayes
- KNN
- Gradient Boosting

### How is the evaluation done?

The evaluation is designed to avoid giving the models information from the same patient in both training and validation.

The protocol is:

- grouping is done by `patient_id`, so studies from the same patient do not leak across train and validation
- splitting is stratified at the group level
- `5-fold x 10 repeats` are used by default
- this gives `50` validation folds per classifier

The grouped split plan is created once and then reused for every classifier. This is important because all models are tested on exactly the same splits, making the comparison fair.

## 5. Select Useful Features Without Data Leakage

When:

`--feature_strategy most_discriminant`

is used, feature selection happens **inside each training fold only**.

The validation fold is never used to decide which features should be selected.

This is important because using validation information during feature selection could make the model look better than it really is.

### Feature-selection process

For each fold:

1. Start with the numerical radiomics features only.
   - metadata such as `patient_id`, `study_id`, `label`, `sample_id`, and PyRadiomics `diagnostics_*` columns are removed
2. Use only the training portion of the current fold.
3. Evaluate each feature individually:
   - invalid or near-constant features are skipped
   - a normality check is attempted
   - if the feature looks Gaussian, a two-sample `t-test` is used
   - otherwise, a `Mann-Whitney U` test is used
   - a univariate ROC AUC is also calculated for ranking
   - the best single-feature threshold is estimated using the Youden index
4. Apply false discovery rate control:
   - Benjamini-Hochberg correction is used
   - features with `q <= fdr_alpha` form the preferred candidate pool
   - if no features survive FDR, the script falls back to the valid ranked features
5. Decide how many features can be kept:
   - the number is not fixed blindly
   - the cap depends on training sample size and minority-class size
   - the goal is to keep the feature set conservative relative to the available data
6. Remove highly redundant features:
   - candidate features are first sorted by univariate relevance
   - a greedy pruning step removes features whose absolute Pearson correlation with an already selected feature is above the threshold
7. Keep the top remaining features up to the inferred cap.
8. Train the classifier using those features and evaluate it on the untouched validation fold.

Because feature selection is repeated independently for every fold, the selected features can be different from one fold to another. This is expected and is the correct behaviour for leakage-safe evaluation.

# What Happens After the `5 x 10` Evaluation?

The repeated cross-validation stage does more than simply produce 50 performance values for each model.

The pipeline saves the predictions and selected features and then performs several additional analyses.

## Fold-Level Results

For every classifier and every fold, the pipeline stores:

- training and validation metrics
- the selected feature subset used in that fold
- validation labels, predictions, and probabilities

## Out-of-Fold Predictions

The predictions from all validation folds are combined into a one-row-per-case table containing:

- classifier
- fold and repeat
- sample, patient, and study identifiers
- true label
- predicted label
- probability of class 1
- selected features for that fold

Because the evaluation is repeated 10 times, the same case can appear in validation more than once.

Therefore, the script combines the repeated out-of-fold predictions by averaging the predicted probability for each case and classifier across all of its validation appearances.

After this aggregation, the script:

- applies the default classification threshold of `0.5`
- creates one aggregated prediction per case and classifier
- calculates patient-level performance summaries

## Bootstrap Confidence Intervals

Using the aggregated out-of-fold predictions, the script performs **stratified bootstrap resampling at the patient level**.

This is used to estimate confidence intervals for:

- AUC
- accuracy
- balanced accuracy
- F1
- MCC
- kappa
- sensitivity
- specificity
- PPV
- NPV

ROC curves with confidence bands are also exported.

## Statistical Comparison Between Models

If:

`--calculate_differences`

is enabled, the script runs:

`train/radiomics/2_modeling/2_model_differences.py`

The comparison works as follows:

- classifiers are compared using the fold-wise metric distributions
- a Friedman global test is applied
- if the global test is significant, pairwise Wilcoxon signed-rank tests are performed with Holm correction

This analysis is used to support the decision about which classifier should move forward to the final optimization stage.

# Final Optimization of the Best Classifier

Script:

`train/radiomics/2_modeling/3_retrain_best_model_and_evaluate.py`

If:

`--fine_tune_best_model`

is enabled, the classifier with the best median validation AUC is retrained in a separate final stage.

The process is:

1. Create a grouped `80/20` train/test split using `GroupShuffleSplit`.
2. Perform feature selection again using only the training split.
3. Use the same training-derived feature subset for both train and test.
4. Optimize the selected classifier using `BayesSearchCV` with grouped cross-validation inside the training split.
5. Save the best estimator.
6. Evaluate the uncalibrated model on the hold-out test split.
7. Estimate test confidence intervals using patient-level bootstrap.
8. Calibrate the predicted probabilities using Platt scaling (`CalibratedClassifierCV`, sigmoid).
9. Evaluate the calibrated model again.
10. Sweep decision thresholds and report the threshold with the best F1.
11. Run SHAP and LIME analyses on both the training split and the hold-out test split.

This final stage provides a more focused evaluation of the selected model and allows its predictions to be analysed and interpreted in greater detail.


# Results and Evaluation

The results below are **reported results from this repository**. They come from the committed result files, logs, predictions, and model outputs in the project. They should not be described as independently verified or independently replicated results.

It is also important to keep the different evaluation protocols separate. The repeated radiomics cross-validation, aggregated out-of-fold results, radiomics holdout experiment, deep-learning five-split evaluation, gland-versus-full comparisons, and the direct radiomics-versus-deep-learning comparison are **different experiments**.

## 1. Dataset

The committed `artifacts/data.csv` contains:

- **MRI examinations:** 1,500
- **Unique patients:** 1,476
- **Positive `case_csPCa = 1`:** 425 (28.33%)
- **Negative `case_csPCa = 0`:** 1,075 (71.67%)
- **Target:** clinically significant prostate cancer (`csPCa`), defined as ISUP grade group `>= 2`
- **MRI modalities:** T2-weighted (T2W), apparent diffusion coefficient (ADC), and high b-value diffusion-weighted imaging (DWI/HBV)

The main radiomics evaluation uses patient-grouped, stratified **5-fold cross-validation repeated 10 times**, giving **50 validation-fold evaluations per classifier**.

Grouping by `patient_id` is used so that studies from the same patient do not appear in both training and validation partitions.

## 2. Feature-Selected Gland Radiomics: Cross-Validation Results

These results come from:

`results/radiomics/most_discriminant/gland/aggregated_performance/summary_metrics.csv`

The values below are the mean and standard deviation across the reported 50 validation-fold evaluations.

| Classifier | Mean validation AUC ± SD | Mean F1 | Mean accuracy | Mean balanced accuracy |
|---|---:|---:|---:|---:|
| Random Forest | **0.8083 ± 0.0229** | 0.5073 | 0.7713 | 0.6650 |
| SVM | 0.8057 ± 0.0229 | **0.6097** | 0.7458 | **0.7332** |
| Gradient Boosting | 0.7959 ± 0.0240 | 0.5327 | 0.7677 | 0.6781 |
| Logistic Regression | 0.7933 ± 0.0259 | 0.5912 | 0.7260 | 0.7189 |
| Naive Bayes | 0.7629 ± 0.0278 | 0.5520 | 0.6739 | 0.6848 |
| KNN | 0.7487 ± 0.0251 | 0.4996 | 0.7491 | 0.6575 |

### What these numbers mean

- **Random Forest** has the highest mean validation AUC: **0.8083 ± 0.0229**.
- **SVM** has the highest mean F1: **0.6097**.
- **SVM** also has the highest mean balanced accuracy: **0.7332**.
- Because the dataset is imbalanced, accuracy should not be considered by itself. Sensitivity, specificity, F1, balanced accuracy, and other metrics are also important.

[Detailed cross-validation results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/gland/aggregated_performance/summary_metrics.csv)

## 3. Aggregated Out-of-Fold Radiomics Results

The repeated cross-validation also produces aggregated **out-of-fold (OOF)** predictions.

These numbers are different from simply averaging the fold-level metrics above. The repository aggregates the repeated validation predictions for each case and then calculates the metrics from those aggregated predictions.

The repository reports **1,500 cases** and **1,476 unique patients** for these results.

All confidence intervals below are reported as **95% confidence intervals**.

### AUC, Accuracy, and F1

| Classifier | OOF AUC (95% CI) | OOF accuracy (95% CI) | OOF F1 (95% CI) |
|---|---:|---:|---:|
| Random Forest | **0.8148 (0.7909–0.8381)** | 0.7807 (0.7619–0.7981) | 0.5280 (0.4832–0.5726) |
| SVM | 0.8082 (0.7840–0.8323) | 0.7693 (0.7490–0.7886) | 0.5411 (0.4987–0.5824) |
| Gradient Boosting | 0.8064 (0.7811–0.8299) | 0.7807 (0.7594–0.7993) | 0.5548 (0.5088–0.5959) |
| Logistic Regression | 0.7977 (0.7718–0.8231) | 0.7267 (0.7040–0.7493) | **0.5933 (0.5602–0.6251)** |
| KNN | 0.7791 (0.7529–0.8029) | 0.7560 (0.7363–0.7759) | 0.5172 (0.4754–0.5587) |
| Naive Bayes | 0.7636 (0.7349–0.7934) | 0.6733 (0.6487–0.6977) | 0.5513 (0.5213–0.5816) |

### Other OOF Metrics

| Classifier | Balanced accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | MCC (95% CI) |
|---|---:|---:|---:|---:|
| Random Forest | 0.6755 (0.6509–0.7008) | 0.4329 (0.3868–0.4811) | **0.9181 (0.9005–0.9348)** | 0.4106 (0.3552–0.4634) |
| SVM | 0.6819 (0.6569–0.7071) | 0.4800 (0.4322–0.5282) | 0.8837 (0.8636–0.9030) | 0.3961 (0.3435–0.4473) |
| Gradient Boosting | 0.6905 (0.6627–0.7157) | 0.4824 (0.4346–0.5327) | 0.8986 (0.8785–0.9171) | 0.4220 (0.3627–0.4722) |
| Logistic Regression | **0.7197 (0.6923–0.7458)** | **0.7035 (0.6604–0.7494)** | 0.7358 (0.7087–0.7630) | 0.4061 (0.3558–0.4543) |
| KNN | 0.6669 (0.6416–0.6924) | 0.4612 (0.4151–0.5083) | 0.8726 (0.8512–0.8925) | 0.3619 (0.3102–0.4158) |
| Naive Bayes | 0.6839 (0.6569–0.7116) | 0.7082 (0.6643–0.7554) | 0.6595 (0.6304–0.6874) | 0.3335 (0.2849–0.3824) |

### Simple interpretation

- Random Forest has the highest OOF AUC: **0.8148 (95% CI 0.7909–0.8381)**.
- Logistic Regression has the highest OOF F1: **0.5933 (95% CI 0.5602–0.6251)**.
- Logistic Regression has much higher OOF sensitivity than Random Forest: **0.7035 vs 0.4329**.
- Random Forest has higher OOF specificity than Logistic Regression: **0.9181 vs 0.7358**.
- Therefore, there is no single model that is automatically “best” for every purpose. The preferred model depends on which type of error matters more.

[OOF AUC and confidence intervals](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/gland/aggregated_performance/auc_ci_summary.txt)

[Full OOF metric table](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/gland/aggregated_performance/summary_metrics.csv)

## 4. Separate Radiomics Hold-Out Experiment

This is a **separate tuned hold-out experiment**. Its numbers should not be combined with the aggregated OOF results above.

The hold-out set contains **303 cases**:

- **218 negatives**
- **85 positives**

### SVM Hold-Out Results

The SVM uses an RBF kernel and **42 selected features**.

Reported settings include:

- `C = 1000`
- `gamma ≈ 0.000308763`
- `coef0 = 1.0`
- FDR alpha `0.05`
- feature correlation threshold `0.9`

| Metric | Reported result |
|---|---:|
| CV AUC | 0.782 ± 0.020 |
| CV F1 | 0.584 ± 0.020 |
| CV balanced accuracy | 0.712 ± 0.018 |
| Holdout AUC, uncalibrated | **0.864** |
| Holdout AUC, bootstrap 95% CI | **0.817–0.906** |
| Holdout AUC, calibrated | **0.865** |
| Calibrated AUC, bootstrap 95% CI | **0.818–0.907** |
| MCC, uncalibrated | 0.554 |
| Cohen's kappa, uncalibrated | 0.548 |
| F1, uncalibrated | 0.688 |
| Accuracy, uncalibrated | 0.805 |
| Sensitivity, uncalibrated | 0.765 |
| Specificity, uncalibrated | 0.821 |
| PPV, uncalibrated | 0.625 |
| NPV, uncalibrated | 0.899 |
| Balanced accuracy, uncalibrated | 0.793 |
| ECE before / after calibration | 0.087 / 0.100 |
| Brier score before / after calibration | 0.137 / 0.143 |

The report also performs a threshold sweep and selects **0.30** based on F1. At this threshold:

- F1 = **0.674**
- Accuracy = **0.799**
- Sensitivity = **0.741**
- Specificity = **0.821**
- PPV = **0.618**
- NPV = **0.891**
- Balanced accuracy = **0.781**
- Calibrated AUC = **0.865**

> **Important:** The threshold `0.30` was selected by sweeping F1 on the hold-out test set itself. Therefore, the metrics at this selected threshold should not be presented as an unbiased final test estimate.

[Full SVM hold-out report](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/gland/best_results/svm/report.txt)

### Logistic Regression Hold-Out Results

| Metric | Reported result |
|---|---:|
| CV AUC | 0.786 ± 0.019 |
| CV F1 | 0.581 ± 0.027 |
| CV balanced accuracy | 0.710 ± 0.018 |
| Holdout AUC, uncalibrated | **0.847** |
| MCC, uncalibrated | 0.553 |
| Cohen's kappa, uncalibrated | 0.545 |
| F1, uncalibrated | 0.688 |
| Accuracy, uncalibrated | 0.802 |
| Sensitivity, uncalibrated | 0.776 |
| Specificity, uncalibrated | 0.812 |
| PPV, uncalibrated | 0.617 |
| NPV, uncalibrated | 0.903 |
| Balanced accuracy, uncalibrated | 0.794 |
| ECE before / after calibration | 0.149 / 0.090 |
| Brier score before / after calibration | 0.162 / 0.143 |

The report chooses threshold **0.30** after a hold-out threshold sweep. At that threshold:

- F1 = **0.681**
- Accuracy = **0.802**
- Sensitivity = **0.753**
- Specificity = **0.821**
- PPV = **0.621**
- NPV = **0.895**
- Balanced accuracy = **0.787**

The same hold-out threshold-selection caveat applies here.

[Full Logistic Regression hold-out report](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/gland/best_results/report.txt)

## 5. Radiomics: Gland Region vs Full Image

This experiment compares radiomics features extracted from:

- the **prostate gland region**
- the **full image**

The values below are **median AUC [IQR]** across the reported paired runs.

| Classifier | Gland median AUC [IQR] | Full-image median AUC [IQR] | Two-sided paired Wilcoxon p-value |
|---|---:|---:|---:|
| Random Forest | **0.8086 [0.0308]** | 0.6083 [0.0396] | 1.7764e-15 |
| Gradient Boosting | **0.8080 [0.0302]** | 0.5905 [0.0356] | 1.7764e-15 |
| SVM | **0.8063 [0.0309]** | 0.6358 [0.0546] | 1.7764e-15 |
| Logistic Regression | **0.7992 [0.0281]** | 0.6204 [0.0394] | 1.7764e-15 |
| KNN | **0.7495 [0.0429]** | 0.5601 [0.0404] | 1.7764e-15 |
| Naive Bayes | **0.7431 [0.0437]** | 0.5651 [0.0397] | 1.7764e-15 |

In this radiomics experiment, all six models have higher reported median AUC when using gland-focused features. The paired tests are also reported as statistically significant.

This result is specific to the **radiomics gland-vs-full experiment**. It should not automatically be applied to the deep-learning comparison.

[Detailed gland-vs-full radiomics results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/tree/main/results/radiomics/most_discriminant/gland_vs_full)

## 6. Radiomics Statistical Comparison Between Classifiers

For the feature-selected gland run, the repository reports:

- **Friedman statistic:** 183.7714
- **Friedman p-value:** 8.3723e-38
- **Significance level:** `alpha = 0.05`

The overall Friedman test indicates statistically significant differences across the classifiers.

For pairwise comparisons using Wilcoxon tests with Holm correction:

- SVM vs Random Forest: adjusted `p = 0.38855`
- Gradient Boosting vs Logistic Regression: adjusted `p = 0.38855`
- Other listed pairwise comparisons in this output are reported as significant.

Other radiomics experiments report:

| Experiment | Friedman statistic | p-value | Repository conclusion |
|---|---:|---:|---|
| Feature-selected gland | 183.7714 | 8.3723e-38 | Differences across classifiers |
| Feature-selected full image | 152.0914 | 4.7894e-31 | Differences across classifiers |
| All-features gland | 216.5600 | 8.1055e-45 | Differences across classifiers |

[Feature-selected gland statistical results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/gland/model_differences/model_differences_summary.txt)

[Feature-selected full-image statistical results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/full/model_differences/model_differences_summary.txt)

[All-features gland statistical results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/all/gland/model_differences/model_differences_summary.txt)

## 7. Deep-Learning Performance on Gland-Focused Input

These values are the mean AUC ± SD across the **five reported splits** in:

`results/deep_learning/model_comparison/simple_statistical_analysis/gland_analysis/csv/model_summary_statistics.csv`

The other columns are mean F1, accuracy, balanced accuracy, sensitivity, and specificity.

`config1`–`config8` are kept as repository labels. Their underlying architecture mappings are not defined here.

| Model | Mean AUC ± SD | AUC min–max | Mean F1 | Mean accuracy | Mean balanced accuracy | Mean sensitivity | Mean specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Config1 | **0.7596 ± 0.0308** | 0.7188–0.8034 | 0.5393 | 0.6947 | 0.6778 | 0.6369 | 0.7187 |
| Config8 | 0.7422 ± 0.0276 | 0.7118–0.7771 | 0.3955 | 0.7360 | 0.6115 | 0.3219 | 0.9011 |
| Base DenseNet | 0.7398 ± 0.0257 | 0.7148–0.7805 | **0.5420** | 0.7239 | 0.6818 | 0.5855 | 0.7781 |
| Config2 | 0.7362 ± 0.0241 | 0.7030–0.7688 | 0.4567 | 0.7121 | 0.6371 | 0.4621 | 0.8120 |
| Base ResNet | 0.7176 ± 0.0335 | 0.6674–0.7531 | 0.4751 | 0.7065 | 0.6347 | 0.4688 | 0.8007 |
| Config7 | 0.6894 ± 0.0494 | 0.6352–0.7600 | 0.3742 | 0.7013 | 0.5878 | 0.3256 | 0.8500 |
| Config6 | 0.6856 ± 0.0475 | 0.6190–0.7301 | 0.3940 | 0.6961 | 0.5927 | 0.3533 | 0.8322 |
| Config4 | 0.6783 ± 0.0183 | 0.6542–0.7052 | 0.3956 | 0.6779 | 0.5883 | 0.3814 | 0.7951 |
| Base EfficientNet-B7 | 0.6724 ± 0.0556 | 0.5945–0.7402 | 0.3191 | 0.6853 | 0.5704 | 0.3067 | 0.8341 |
| Config5 | 0.6583 ± 0.0310 | 0.6159–0.6890 | 0.4512 | 0.6458 | 0.6118 | 0.5351 | 0.6885 |
| Base EfficientNet | 0.6436 ± 0.0261 | 0.6066–0.6765 | 0.3632 | 0.6941 | 0.5773 | 0.3077 | 0.8468 |
| Base EfficientNet-B8 | 0.6288 ± 0.0456 | 0.5608–0.6835 | 0.3385 | 0.6547 | 0.5539 | 0.3219 | 0.7859 |
| Config3 | 0.5998 ± 0.1396 | 0.4153–0.7621 | 0.3869 | 0.5364 | 0.5658 | 0.6381 | 0.4935 |
| Base ViT | 0.5000 ± 0.0000 | 0.5000–0.5000 | 0.4416 | 0.2834 | 0.5000 | 1.0000 | 0.0000 |

### Simple interpretation

- **Config1** has the highest reported mean AUC: **0.7596**.
- Base DenseNet has mean AUC **0.7398** and mean F1 **0.5420**.
- Config8 has high mean specificity (**0.9011**) but low mean sensitivity (**0.3219**).
- Base ViT has AUC **0.5** and specificity **0.0** across these splits. This pattern is consistent with degenerate class predictions rather than useful discrimination.

[Deep-learning model results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/deep_learning/model_comparison/simple_statistical_analysis/gland_analysis/csv/model_summary_statistics.csv)

### Deep-Learning Statistical Test

The repository's AUC statistical analysis reports:

- **Friedman statistic:** 43.7106
- **Friedman p-value:** 3.4268e-05
- `alpha = 0.05`
- Overall Friedman result: statistically significant differences across the models.
- No pairwise comparison remained statistically significant after multiple-comparison correction.

[Deep-learning statistical analysis](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/deep_learning/model_comparison/simple_statistical_analysis/gland_analysis/statistical_analysis/statistical_analysis_test_auc.txt)

## 8. Deep Learning: Gland ROI vs Full Image

The following values are **median AUC [IQR]**. The p-value is from the report's two-sided paired Wilcoxon test.

| Model | Gland ROI AUC [IQR] | Full-image AUC [IQR] | Two-sided p-value |
|---|---:|---:|---:|
| Base DenseNet | 0.7374 [0.0234] | 0.6748 [0.0233] | 0.0625 |
| Base EfficientNet | 0.6420 [0.0225] | 0.6324 [0.0130] | 1.0000 |
| Base EfficientNet-B7 | 0.6628 [0.0536] | 0.6366 [0.0353] | 0.0625 |
| Base EfficientNet-B8 | 0.6379 [0.0354] | 0.6270 [0.0372] | 1.0000 |
| Base ResNet | 0.7297 [0.0338] | 0.6019 [0.0318] | 0.0625 |
| Base ViT | 0.5000 [0.0000] | 0.5586 [0.0215] | 0.0625 |
| Config1 | **0.7533 [0.0185]** | 0.7127 [0.0423] | 0.0625 |
| Config2 | 0.7415 [0.0156] | 0.6445 [0.0181] | 0.0625 |
| Config3 | 0.6503 [0.1711] | 0.5349 [0.0159] | 0.3125 |
| Config4 | 0.6755 [0.0067] | 0.6239 [0.0379] | 0.0625 |
| Config5 | 0.6524 [0.0425] | 0.6226 [0.0170] | 0.3125 |
| Config6 | 0.6947 [0.0697] | 0.6201 [0.0270] | 0.1250 |
| Config7 | 0.7009 [0.0512] | 0.6276 [0.0276] | 0.0625 |
| Config8 | 0.7356 [0.0415] | 0.6688 [0.0568] | 0.0625 |

The gland-focused median AUC is numerically higher for **13 of the 14 models**, with ViT being the exception.

However, none of the listed two-sided p-values is below **0.05**, so this comparison does **not** establish a statistically significant gland-versus-full difference for the deep-learning models.

[Deep-learning gland-vs-full results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/tree/main/results/deep_learning/model_comparison/simple_statistical_analysis/gland_vs_full)

## 9. Direct Radiomics vs Deep-Learning Comparison

This comparison comes from:

`results/compare_best_radiomics_dl/summary.txt`

It compares **Logistic Regression radiomics** with deep-learning **config1** over **five folds**.

This is a separate experiment. It is not the same as the 50-fold radiomics OOF results or the five-split deep-learning summary above.

| Fold | Deep-learning AUC | Radiomics AUC |
|---:|---:|---:|
| 1 | 0.803 | 0.802 |
| 2 | 0.719 | 0.794 |
| 3 | 0.753 | 0.784 |
| 4 | 0.770 | 0.799 |
| 5 | 0.752 | 0.832 |

The summary reports:

- Mean radiomics AUC from the displayed rounded values: approximately **0.802**
- Mean deep-learning AUC from the displayed rounded values: approximately **0.759**
- Wilcoxon signed-rank statistic: **1.0000**
- Wilcoxon p-value: **0.1250**
- Paired Cohen's `d`: **-1.227** (as printed in the file)
- Repository conclusion: **no significant difference was observed across these five folds**

### What should be concluded?

Radiomics has the higher numerical AUC in these five folds, but this small comparison does **not** establish statistical superiority.

In particular, the reported **p = 0.1250** means that we should **not claim that radiomics statistically outperforms deep learning** based on this comparison.

If the paired Cohen's `d` is quoted, its sign should be preserved exactly as reported. It should not be over-interpreted without checking the comparison script and its sign convention.

[Direct radiomics-vs-DL comparison](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/compare_best_radiomics_dl/summary.txt)

## 10. Additional Radiomics Experiments

The repository also contains separate experiments that should not be mixed with the feature-selected gland results.

### All-Features Gland Radiomics

This run is located under:

`results/radiomics/all/gland/`

Mean validation AUC ± SD across its 50 fold evaluations:

| Classifier | Mean validation AUC ± SD |
|---|---:|
| Gradient Boosting | **0.8069 ± 0.0235** |
| SVM | 0.8047 ± 0.0240 |
| Random Forest | 0.8002 ± 0.0233 |
| Logistic Regression | 0.7570 ± 0.0265 |
| KNN | 0.6904 ± 0.0302 |
| Naive Bayes | 0.6646 ± 0.0293 |

Statistical result:

- Friedman statistic: **216.5600**
- Friedman p-value: **8.1055e-45**
- The report concludes that there are overall statistically significant differences across classifiers.

[All-features gland results](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/all/gland/resultados_features_all_gland_all.csv)

[All-features gland statistical test](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/all/gland/model_differences/model_differences_summary.txt)

### Feature-Selected Full-Image Radiomics

Mean validation AUC ± SD:

| Classifier | Mean validation AUC ± SD |
|---|---:|
| SVM | **0.6312 ± 0.0353** |
| Logistic Regression | 0.6163 ± 0.0370 |
| Random Forest | 0.6098 ± 0.0299 |
| Gradient Boosting | 0.5881 ± 0.0292 |
| Naive Bayes | 0.5681 ± 0.0312 |
| KNN | 0.5577 ± 0.0313 |

Statistical result:

- Friedman statistic: **152.0914**
- Friedman p-value: **4.7894e-31**

[Feature-selected full-image fold metrics](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/full/resultados_features_all_full_most_discriminant.csv)

[Feature-selected full-image statistical test](https://github.com/BIMCV-CSUSP/Radiomics-Prostate-Cancer/blob/main/results/radiomics/most_discriminant/full/model_differences/model_differences_summary.txt)

## 11. Important Limitations

These results should be presented with the following points in mind:

- **Repository-reported, not independently replicated:** the numbers are committed in this repository. They should not be described as independently verified unless an independent reproduction is documented.
- **Class imbalance:** 425/1,500 examinations (**28.3%**) are positive and 1,075/1,500 (**71.7%**) are negative. Sensitivity and specificity should therefore be considered along with AUC and accuracy.
- **Different evaluation protocols:** fold-level means, aggregated OOF scores, holdout results, deep-learning five-split summaries, gland/full comparisons, and direct radiomics/DL comparisons are different experiments. Tables should always state which protocol produced the numbers.
- **Holdout threshold selection:** the SVM and Logistic Regression reports sweep F1 on the holdout data and select threshold `0.30`. Metrics at that selected threshold may therefore be optimistically biased. Ideally, the threshold should be fixed using training/validation data and then evaluated once on a truly untouched test set.
- **Statistical significance:** the direct five-fold radiomics-versus-deep-learning comparison reports `p = 0.125`, so it does not support a claim that radiomics statistically outperforms deep learning.
- **Multiple result files:** the repository contains multiple result files and older experimental directories, including `z_old_k=10`. The named current summaries should be preferred, and the exact experiment/configuration should be stated whenever values differ.
- **No clinical-readiness claim:** these are retrospective experimental model results. They should not be presented as evidence of clinical validation or deployment readiness.

## 12. Quick Results Summary

For readers who do not need every metric, the main reported results can be summarized as follows:

| Experiment | Headline result | What it means |
|---|---|---|
| Feature-selected gland radiomics, repeated CV | Random Forest mean fold AUC **0.8083 ± 0.0229**; aggregated OOF AUC **0.8148 (95% CI 0.7909–0.8381)** | Random Forest has the highest OOF AUC in the reported six-classifier radiomics table, but its OOF sensitivity is 0.4329 |
| Feature-selected gland radiomics, SVM holdout | AUC **0.864 (95% CI 0.817–0.906)** | Separate internal holdout experiment; threshold tuning on the holdout set is an important caveat |
| Deep learning on gland ROI | Config1 mean AUC **0.7596 ± 0.0308** across five reported splits | Highest mean AUC in the committed five-split deep-learning summary |
| Radiomics vs deep learning | Five-fold rounded mean AUC ≈ **0.802 vs ≈ 0.759**; Wilcoxon `p = 0.125` | Radiomics is numerically higher in this comparison, but the difference is not statistically significant |
| Radiomics gland vs full image | Gland median AUC around **0.743–0.809** vs full-image AUC around **0.560–0.636** | Gland-focused radiomics performs better in this specific comparison; the reported paired tests are significant |
| Deep-learning gland vs full image | Gland median AUC is higher for **13/14 models** | This is a numerical trend only; none of the listed two-sided tests is significant at 0.05 |

This summary is **not a single unified benchmark**. Each row comes from a different evaluation protocol.


# Why These Evaluation Choices Matter

Several parts of the pipeline are specifically designed to make the results more trustworthy.

### Patients are kept together

A patient's data should not appear in both training and validation. Grouping by `patient_id` prevents this type of leakage.

### Feature selection is done inside each fold

The validation data is not used to select features. This gives a more realistic estimate of how the model performs on unseen data.

### Every classifier uses the same folds

The same grouped fold plan is reused across classifiers. This makes model-to-model comparisons fair.

### Predictions are combined at the case level

Because the cross-validation is repeated, each case can have multiple validation predictions. These are combined by averaging the predicted probabilities.

### Confidence intervals are estimated at the patient level

Bootstrap resampling is done at the patient level so that repeated studies from the same patient are not treated as completely independent samples.

# Reproducibility Notes

The current implementation includes several choices aimed at making the experiments reproducible:

- grouped splitting by patient
- fold-wise feature selection to avoid leakage
- shared fold plans across classifiers for fair comparison
- exported selected features per fold
- aggregated out-of-fold predictions at the case level
- bootstrap confidence intervals at the patient level
- project-root-based path resolution instead of fragile relative paths

### Important methodological note

In the current final hold-out script, the threshold sweep is performed on the hold-out test set itself.

This is useful for exploratory analysis, but if the threshold is intended to be fixed for a final unbiased evaluation, it should instead be selected using a separate validation layer within the training data, rather than using the test split.

# Typical Commands

## Build the concatenated radiomics table

```bash
python train/radiomics/2_modeling/0_build_concatenated_feature_table.py \
  --radiomics_root artifacts/radiomics \
  --mode gland \
  --keep_shape_from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv
```

## Run the main radiomics benchmark

```bash
python train/radiomics/2_modeling/1_train_and_evaluate.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --ci_level 0.95 \
  --classification_threshold 0.5 \
  --min_features 10 \
  --max_features_cap 60 \
  --samples_per_feature 25 \
  --minority_samples_per_feature 8 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.90 \
  --selection_n_jobs 8 \
  --search_n_jobs 8 \
  --search_iterations 50 \
  --calculate_differences \
  --fine_tune_best_model
```

## Run the final hold-out optimization directly

```bash
python train/radiomics/2_modeling/3_retrain_best_model_and_evaluate.py \
  --csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --model LogisticRegression \
  --feature_strategy most_discriminant \
  --bootstrap_iterations 1000 \
  --ci_level 0.95
```

# Related Modules

- `train/radiomics/README.md`: more detailed information about the radiomics methodology
- `train/deep_learning/README.md`: information about the deep learning branch
- `train/compare_approaches/`: scripts for directly comparing radiomics and deep learning approaches

# Reference

[1] A. Saha, J. S. Bosma, J. J. Twilt, B. van Ginneken, A. Bjartell, A. R. Padhani, D. Bonekamp, G. Villeirs, G. Salomon, G. Giannarini, J. Kalpathy-Cramer, J. Barentsz, K. H. Maier-Hein, M. Rusu, O. Rouvière, R. van den Bergh, V. Panebianco, V. Kasivisvanathan, N. A. Obuchowski, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, and the PI-CAI consortium. “Artificial Intelligence and Radiologists in Prostate Cancer Detection on MRI (PI-CAI): An International, Paired, Non-Inferiority, Confirmatory Study”. *The Lancet Oncology* 2024; 25(7): 879-887.
