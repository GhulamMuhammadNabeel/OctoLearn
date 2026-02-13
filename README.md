# OctoLearn 🐙

**Structured AutoML Pipeline with Intelligent Dataset Profiling**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OctoLearn generates **professional-grade intelligence dossiers** on your datasets automatically. In under 1 second, you get:

- 📊 **Risk Score** (0-100 data quality assessment)
- 📈 **Feature Importance** (baseline model + SHAP analysis)
- 🔧 **Preprocessing Strategy** (imputation, encoding, scaling recommendations)
- 📉 **Visual Analysis** (distributions, correlations, SHAP plots)
- 💡 **Strategic Recommendations** (machine learning insights)
- 📄 **Professional PDF Report** (ready for stakeholders)

No code. No notebooks. Just intelligence.

---

## ⚡ Quick Start

### Installation

```bash
pip install octolearn
```

### Basic Usage

```python
from octolearn import AutoML
from sklearn.datasets import load_iris

# Load your dataset
data = load_iris(as_frame=True)
X, y = data.data, data.target

# Initialize and fit
automl = AutoML()
automl.fit(X, y)

# Generate comprehensive report
pdf_file = automl.generate_report()
print(f"Report saved: {pdf_file}")
# Output: octolearn_report_8145f023f195.pdf
```

### Access Individual Analyses

```python
# Get dataset risk score
risk = automl.get_risk_score()
print(f"Risk Score: {risk['score']}/100")
print(f"Category: {risk['category']}")

# Get feature importance
importance = automl.get_feature_importance()
for feature, score in list(importance.items())[:5]:
    print(f"{feature}: {score:.4f}")

# Get preprocessing suggestions
suggestions = automl.get_preprocessing_suggestions()
for strategy, recs in suggestions.items():
    print(f"{strategy}: {recs}")
```

---

## 🎯 Features

### 1. Dataset Intelligence (16 Metrics)
- Row/column count
- Feature type detection (numeric, categorical, datetime)
- Missing value analysis
- Duplicate detection
- Cardinality assessment
- Skewness detection
- Task type auto-detection

### 2. Risk Scoring (0-100)
Comprehensive data quality assessment including:
- Missing data impact (0-20 points)
- Duplicate rows (0-15 points)
- Class imbalance (0-15 points)
- Feature skewness (0-10 points)
- Constant columns (0-10 points)
- High cardinality features (0-10 points)
- Feature-to-sample ratio (0-10 points)
- Sample size penalty (0-5 points)

**Risk Categories:**
- ✅ 0-30: Low Risk
- ⚠️ 31-60: Moderate Risk
- ❌ 61-100: High Risk

### 3. Preprocessing Recommendations
Auto-generated strategy for:
- **Missing Values**: Mean, median, KNN, or iterative imputation
- **Categorical Encoding**: One-Hot, Ordinal, or Target Encoding
- **Feature Scaling**: StandardScaler vs RobustScaler vs MinMaxScaler
- **Feature Engineering**: Polynomial features, interactions, temporal features
- **Column Actions**: Removal of constants, cardinality handling

### 4. Feature Importance
- **Baseline Model**: Fast Random Forest (~150ms)
- **SHAP Analysis**: TreeExplainer Shapley values
- **Ranking**: Top features identified automatically

### 5. Visual Analysis
6 professional visualizations embedded in PDF:
- Feature distributions (histograms with KDE)
- Correlation heatmap
- SHAP feature importance bar chart

### 6. Professional Reports
177KB PDF with:
- Color-coded risk assessment
- Executive summary
- Detailed diagnostics
- Visual analysis
- Actionable recommendations
- Professional formatting

---

## 📊 Example Output

```
DATASET PROFILE:
   Rows: 150
   Columns: 4
   Task Type: classification
   Hash: 8145f023f195

RISK SCORE: 10/100 (Low Risk)
   Risk Factors:
   • 0.7% duplicate rows
   • Features/samples ratio: 0.03

FEATURE IMPORTANCE (Top 3):
   1. petal length (cm): 0.4442
   2. petal width (cm): 0.4181
   3. sepal length (cm): 0.1099

PREPROCESSING SUGGESTIONS:
   Missing Values: No missing values detected
   Categorical Encoding: No categorical features
   Scaling: Use StandardScaler for tree models
   Feature Engineering: Consider polynomial features
```

---

## 🔬 API Reference

### `AutoML()`

Main orchestrator class.

#### Methods

##### `fit(X, y)`
Train profiler on dataset.
- **Parameters**: X (DataFrame), y (Series or array)
- **Returns**: self

##### `generate_report()`
Create comprehensive PDF report.
- **Returns**: filename (str)
- **Output**: Saves PDF with hash-based name

##### `get_risk_score()`
Get data quality assessment without full report.
- **Returns**: `{"score": int, "category": str, "factors": dict}`

##### `get_preprocessing_suggestions()`
Get preprocessing strategy without full report.
- **Returns**: dict with 5 recommendation categories

##### `get_feature_importance()`
Get baseline feature importance without full report.
- **Returns**: dict of `{feature: score}`

##### `report()`
Get raw profile dataclass.
- **Returns**: DatasetProfile

---

## 🏗 Architecture

```
octolearn/
├── core.py                      ← Main AutoML class
├── config.py                    ← Configuration
├── profiling/
│   └── data_profiler.py        ← Dataset analysis (16 metrics)
├── experiments/
│   ├── risk_scorer.py          ← Risk assessment (0-100)
│   ├── preprocessing_suggester.py  ← Preprocessing strategy
│   ├── baseline_importance.py   ← Feature ranking
│   ├── plot_generator.py        ← Visualization + SHAP
│   ├── recommendation_engine.py ← Strategic insights
│   └── report_generator.py      ← PDF factory
├── [feature, models, optimization, evaluation, utils]  ← Future layers
```

---

## ⚡ Performance

| Task | Time |
|------|------|
| Profile dataset | ~50ms |
| Calculate risk score | ~30ms |
| Generate suggestions | ~20ms |
| Train baseline model | ~150ms |
| Create visualizations | ~200ms |
| Generate PDF | ~100ms |
| **TOTAL** | **~550ms** |

---

## 📋 Requirements

- Python 3.8+
- pandas ≥ 1.0.0
- numpy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- reportlab ≥ 3.6.0
- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0
- shap ≥ 0.40.0

---

## 🚀 Roadmap

### Phase 2 ✅ Complete
- [x] Dataset profiling
- [x] Risk scoring
- [x] Preprocessing suggestions
- [x] Feature importance
- [x] SHAP analysis
- [x] PDF generation

### Phase 3 (Coming Soon)
- [ ] Model performance prediction
- [ ] Outlier detection & visualization
- [ ] Feature interaction analysis
- [ ] Automatic data cleaning
- [ ] Cross-validation strategy recommendation
- [ ] Ensemble model selection
- [ ] MLflow integration

### Phase 4 (Roadmap)
- [ ] AutoML model training
- [ ] Hyperparameter optimization
- [ ] Automated pipeline building
- [ ] Production model deployment

---

## 🧪 Testing

### Run Tests

```bash
# Basic test
python test_octolearn.py

# Validation test
python validation.py

# Interactive notebook
jupyter notebook octolearn_demo.ipynb
```

### Example Notebooks

- `octolearn_demo.ipynb`: Interactive feature showcase

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

---

## 📞 Support

**Issues or Questions?**

- Check documentation in [ARCHITECTURE.md](ARCHITECTURE.md)
- Review [octolearn_demo.ipynb](octolearn_demo.ipynb)
- Open an issue on GitHub

---

## 🐙 Philosophy

> "Build the skeleton first. Then make it breathe."

OctoLearn follows modular, automated principles:
- ✅ **Modular Architecture** - Each component independent
- ✅ **Automation First** - Recommendations, not questions
- ✅ **Professional Quality** - Production-ready outputs
- ✅ **Reproducible Results** - Hash-based naming
- ✅ **Zero Configuration** - Works out of the box

---

**Made with ☕ and 🐙 Logic.**

*OctoLearn v0.2.0 — Intelligent AutoML for Everyone*
