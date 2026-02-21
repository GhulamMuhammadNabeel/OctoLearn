# OctoLearn User Guide

Welcome to the comprehensive guide for OctoLearn. This document provides in-depth information on how to configure and extend the AutoML pipeline.

## 📖 Table of Contents
- [Core Concepts](#core-concepts)
- [Configuration Reference](#configuration-reference)
- [Advanced Features](#advanced-features)
- [Methodology](#methodology)
- [Troubleshooting](#troubleshooting)

---

## Core Concepts

OctoLearn is built on the principle of **Transparent AutoML**. Unlike other libraries that hide the "magic" inside a black box, OctoLearn exposes the results of every stage—from raw data profiling to final model staging.

### The fit-then-report Pattern
Success with OctoLearn follows a simple pattern:
1.  **fit()**: Orchestrates the technical pipeline (cleaning, tuning, training).
2.  **generate_report()**: Decodes the technical results into human-readable business intelligence.

---

## Configuration Reference

You can pass specific configuration objects to the `AutoML` constructor to control behavior.

### DataConfig
Controls how data is sampled and split.
- `use_full_data`: If True, bypasses sampling.
- `test_size`: Fraction of data held out for validation (default 0.2).
- `random_state`: Integer seed for reproducibility.

### PreprocessingConfig
Controls the automated data cleaning engine.
- `imputer_strategy`: `{'numeric': 'median', 'categorical': 'mode'}`.
- `scaler`: `'standard'`, `'robust'`, or `'minmax'`.

---

## Advanced Features

### Class Imbalance Handling
OctoLearn automatically detects class imbalance during profiling and utilizes stratified splitting to ensure stable evaluation metrics.

### Target Leakage Detection
The `DataProfiler` looks for features that are essentially identical to or highly correlated with the target, flagging them as potential "leakage suspects" in the risk report.

---

## Methodology

### Stacking Ensembles
When multiple models are selected for training, OctoLearn generates a Stacking Regressor or Classifier by using the top-performing base models and a meta-model to blend their predictions.

---

## Troubleshooting

### PDF Generation Fails
Ensure you have `reportlab` installed. If your environment lacks specific fonts, OctoLearn will fallback to standard Helvetica.

### Optuna is Too Slow
Reduce `optuna_trials` or set `use_optuna=False` in your `fit()` call for a baseline performance run.
