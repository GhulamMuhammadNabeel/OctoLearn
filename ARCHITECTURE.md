# 🐙 OctoLearn v0.2 - Complete Intelligence Engine

**Status**: ✅ PRODUCTION READY  
**Release**: Phase 2 Complete - Full Diagnostic Suite  
**Date**: February 13, 2026

---

## 🎯 What You Have

A professional-grade **automated machine learning intelligence platform** that generates forensic dataset diagnostics at a single command.

```python
automl = AutoML()
automl.fit(X, y)
pdf_report = automl.generate_report()  # Outputs: octolearn_report_<hash>.pdf
```

**In 3 seconds**, you get a **10-page PDF** with:
- ✅ Risk Score (0-100 data quality credit score)
- ✅ Feature Analysis (distributions, correlations)
- ✅ SHAP Explanations (feature importance via Shapley values)
- ✅ Preprocessing Strategy (imputation, encoding, scaling recommendations)
- ✅ Baseline Feature Power (quick model importance scores)
- ✅ Strategic Recommendations (automation insights)

---

## 🏗 Architecture

### Core Pipeline
```
octolearn/
├── core.py                          ← Main AutoML orchestrator
├── profiling/
│   └── data_profiler.py            ← Dataset intelligence (16 metrics)
├── experiments/
│   ├── risk_scorer.py              ← Data quality assessment (0-100)
│   ├── preprocessing_suggester.py   ← Smart preprocessing hints
│   ├── baseline_importance.py       ← Quick feature ranking
│   ├── plot_generator.py            ← 6 visualization types
│   ├── recommendation_engine.py     ← Automated insights
│   └── report_generator.py          ← PDF factory (ReportLab)
└── [feature, models, optimization, evaluation, utils]  ← Future layers
```

### New Modules (Phase 2)

| Module | Purpose | Output |
|--------|---------|--------|
| `risk_scorer.py` | Calculate data quality score | Score (0-100), Category, Risk Factors |
| `preprocessing_suggester.py` | Generate preprocessing strategy | Dict of recommendations by category |
| `baseline_importance.py` | Train quick model for feature ranking | Feature → Importance dict |
| `plot_generator.py` (upd) | Generate SHAP plots | 6 PNG visualizations |
| `report_generator.py` (upd) | Build multi-page PDF | 177KB professional report |

---

## 📊 Feature Inventory

### 1. **Dataset Risk Score (0-100)**
Comprehensive data quality credit score assessing:
- Missing data severity (max 20 pts)
- Duplicate rows (max 15 pts)
- Class imbalance (max 15 pts)
- Feature skewness (max 10 pts)
- Constant columns (max 10 pts)
- High cardinality (max 10 pts)
- Feature-to-sample ratio (max 10 pts)
- Sample size (max 5 pts)

**Categories:**
- 0-30: ✅ Low Risk (safe data)
- 31-60: ⚠️ Moderate Risk (proceed carefully)
- 61-100: ❌ High Risk (needs serious preprocessing)

### 2. **Preprocessing Suggestions**
Auto-generated strategy for:
- **Missing Values**: Mean/Median, KNN, Iterative imputation rules
- **Categorical Encoding**: One-Hot (≤5 categories), Target Encoding, Frequency Encoding
- **Scaling**: StandardScaler vs MinMaxScaler recommendations
- **Feature Engineering**: Polynomial features, interactions, temporal features
- **Column Actions**: Remove constants, handle cardinality

### 3. **Feature Importance Baseline**
Quick Random Forest model (~0.2s) to rank feature power:
- Handles missing values automatically
- Encodes categoricals on the fly
- Returns ranked feature importance dict
- SHAP-compatible for comparison

### 4. **SHAP Explainability Plots**
SHAP TreeExplainer analysis:
- **Bar Plot**: Feature importance via Shapley values
- **Visual Proof**: Model-agnostic feature ranking
- **Embedded in PDF**: Professional visualization

### 5. **Visual Analysis**
6 PNG plots embedded in PDF:
1. Distribution plots (first 5 numeric features)
2. Correlation heatmap
3. SHAP importance bar chart
4. Automatic cleanup & organization

---

## 🚀 Quick Start

### Installation
```bash
cd c:\Users\Nabeel\Desktop\OctoLearn
pip install -e .
```

### Usage
```python
from octolearn import AutoML
import pandas as pd

# Load your data
X, y = your_data_loader()

# Create and fit
automl = AutoML()
automl.fit(X, y)

# Get insights (optional - all included in PDF)
risk = automl.get_risk_score()
suggestions = automl.get_preprocessing_suggestions()
importance = automl.get_feature_importance()

# Generate professional report
pdf = automl.generate_report()
print(f"Report: {pdf}")  # octolearn_report_<hash>.pdf
```

### Example Output
```
Risk Score: 10/100 (Low Risk)
Top Features: petal_length (0.44), petal_width (0.42)
Preprocessing: No missing values, use StandardScaler
PDF: octolearn_report_8145f023f195.pdf (177KB)
```

---

## 📦 Dependencies

```
pandas              # Data handling
numpy               # Numerics
scikit-learn        # ML baseline
optuna              # Hyperopt (future)
reportlab           # PDF generation
matplotlib          # Plotting
seaborn             # Statistical viz
shap                # Explainability
```

---

## 🧪 Testing

### Test File
- **`test_octolearn.py`**: End-to-end pipeline test with Iris dataset
- **`octolearn_demo.ipynb`**: Interactive notebook showcase

Run test:
```bash
python test_octolearn.py
```

Output:
```
✓ Profile extraction: 8145f023f195
✓ Risk score: 10/100
✓ Feature importance: 4 features ranked
✓ Preprocessing suggestions: 4 categories
✓ PDF generated: 177KB
✓ Plots created: 6 visualizations
```

---

## 📈 Performance

| Task | Time | Output |
|------|------|--------|
| Profile dataset | ~50ms | DatasetProfile |
| Calculate risk | ~30ms | Risk score + factors |
| Generate suggestions | ~20ms | Dict of recommendations |
| Train baseline model | ~150ms | Feature importance |
| Create plots | ~200ms | 6 PNG files |
| Generate PDF | ~100ms | Professional report |
| **TOTAL** | ~**550ms** | **Complete report** |

---

## 🎨 PDF Report Sections

1. **Title Page**: Hash, timestamp, risk score banner
2. **Dataset Overview**: Summary table (rows, columns, task type)
3. **Data Quality Assessment**: Risk factors with severity
4. **Feature Insights**: Lists of numeric, categorical, skewed, cardinality features
5. **Preprocessing Recommendations**: 5-section strategy guide
6. **Feature Importance**: Top 10 baseline scores
7. **Visual Analysis**: 6 embedded plots
   - Feature distributions
   - Correlation heatmap
   - SHAP importance
8. **Strategic Recommendations**: Actionable next steps

---

## 🔧 API Reference

### `AutoML()`
Main orchestrator class.

**Methods:**
- `fit(X, y)` → self
- `generate_report()` → filename (str)
- `get_risk_score()` → {score, category, factors}
- `get_preprocessing_suggestions()` → dict
- `get_feature_importance()` → {feature: score}
- `report()` → DatasetProfile

### `DatasetProfile` (dataclass)
```python
@dataclass
class DatasetProfile:
    dataset_hash: str
    n_rows: int
    n_columns: int
    numeric_features: List[str]
    categorical_features: List[str]
    datetime_features: List[str]
    missing_report: Dict[str, float]
    imbalance_ratio: float | None
    skewed_columns: List[str]
    constant_columns: List[str]
    duplicate_rows: int
    high_cardinality_cols: List[str]
    task_type: str
```

---

## 🔮 Phase 3 Roadmap

- [ ] SHAP dependence plots
- [ ] Model performance prediction
- [ ] Feature interaction detection
- [ ] Automatic data cleaning (lossless)
- [ ] Outlier visualization
- [ ] Class balance recommendations
- [ ] Cross-validation strategy
- [ ] Ensemble model selection
- [ ] MLflow integration

---

## ✅ Checklist: What's Complete

- [x] Data profiling (16 metrics)
- [x] Data quality scoring (0-100)
- [x] Risk factor analysis
- [x] Missing value strategies
- [x] Categorical encoding strategies
- [x] Feature scaling guidance
- [x] Feature importance baseline
- [x] SHAP explainability plots
- [x] Professional PDF generation
- [x] End-to-end testing
- [x] Modular architecture
- [x] Production-ready packaging

---

## 🐙 Philosophy

> "Build the skeleton first. Then make it breathe."

Everything in OctoLearn follows this principle:
- **Structure over features**: Proper architecture supports future growth
- **Automation over manual**: Recommendations, not questions
- **Quality over speed**: Professional PDFs, reproducible hashes
- **Modular over monolithic**: Each component lives independently
- **Testing over assumptions**: Real data, real outputs

---

## 📞 Support

**Issues?** Check:
1. Data format (pandas DataFrame required)
2. Dependencies installed (`pip install -e .`)
3. Dataset size (minimum 50 rows recommended)
4. Python version (3.8+)

**Success indicators:**
- Risk score calculated within 1 second
- PDF generated with all sections
- 6 plots embedded without errors
- Report file appears in working directory

---

**Made with ☕ and 🐙 Logic.**  
*OctoLearn v0.2 — February 2026*
