# Deep Learning: File Structure and Methodology

## File Structure

```text
1_modeling/
├── config.json                              # Training parameters
├── train.py                                 # Main training script
└── data_loaders/                            # Data-loading scripts used during training
    ├── data_loader_for_cv_org.py
    └── data_loader_for_cv_roi.py

2_analyse_results/
├── predict_&_analyse_probs/
│   ├── 1_predict.py                         # Generate predictions
│   ├── 2_analyse_predictions.py             # Analyse the distribution of predictions
│   └── z_data_loader_for_cv_for_predict.py  # Load data for evaluation
├── simple_statistical_analysis/
│   └── compare_models.py                    # Statistical comparison of model metrics

3_model_explicability/
└── explain_predictions.py                   # Analyse model predictions and explainability
```

## Methodology

The Deep Learning analysis is organized into three main stages:

### 1. Training and Validation — [`train.py`](./1_modeling/train.py)

This script trains convolutional models using stratified group cross-validation.

#### 1. Data Loading

Two data-loading strategies are available, depending on the value of the `--mode` parameter:

- [`data_loader_for_cv_org.py`](./1_modeling/data_loaders/data_loader_for_cv_org.py): uses the complete image as input.
- [`data_loader_for_cv_roi.py`](./1_modeling/data_loaders/data_loader_for_cv_roi.py): uses only the region corresponding to the prostate gland.

#### 2. Models and Configurations

- The models to be used are defined in `config.json`, with each key representing a specific configuration.
- Models can be defined with or without additional transformations using the `extra_transforms` field.

#### 3. Training and Evaluation

- The selected model is trained using cross-validation, with standard performance metrics including AUC, F1 (macro and binary), accuracy, sensitivity, specificity, MCC, and others.
- Early stopping is applied based on validation AUC.
- The best model from each split is saved, along with the overall best-performing model.

#### Generated Files

The results from this stage are stored in [`artifacts/deep_learning`](../../artifacts/deep_learning/), organized by input mode (`full` or `gland`) and model configuration.

The generated files include:

- Model checkpoints for each split and the overall best model (`.pth`)
- `.csv` files containing training and validation metrics
- Training logs in `.log` format

### 2. Model Comparison

Two complementary approaches are used to compare the performance of the trained models:

#### 1. Direct Comparison

- The [`compare_models.py`](./2_analyse_results/simple_statistical_analysis/compare_models.py) script analyses the training results directly from the generated files.
- Promising models are identified by comparing metrics such as AUC, F1, accuracy, sensitivity, and others.
- Statistical tests are applied, including the Friedman test and post-hoc comparisons using the Wilcoxon test with Holm correction.

#### 2. Prediction-Based Comparison

- The [`1_predict.py`](./2_analyse_results/predict_&_analyse_probs/1_predict.py) script generates predictions from the trained models on their validation sets, including the probability assigned to each class for every patient.
- These probabilities are compared across models using [`2_analyse_predictions.py`](./2_analyse_results/predict_&_analyse_probs/2_analyse_predictions.py), which performs statistical analysis along with visualizations such as boxplots and p-value heatmaps.

#### Generated Files

The results are stored in [`results/deep_learning/model_comparison`](../../results/deep_learning/model_comparison/), including:

- Metric visualizations such as radar charts, bar charts, and boxplots
- `.csv` files containing combined metrics and summary statistics
- `.txt` reports containing the results of the statistical analyses

### 3. Model Explainability

The [`explain_predictions.py`](./3_model_explicability/explain_predictions.py) script is used for analysing model predictions and their explainability.

<!-- TO-DO -->
