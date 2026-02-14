import pandas as pd
from sklearn.datasets import load_diabetes
from octolearn import AutoML
import seaborn as  sns
# data = load_diabetes(as_frame=True)
data=sns.load_dataset('titanic')
X = data.drop('survived', axis=1)
y = data['survived']

automl = AutoML(show_progress=False)
automl.fit(X, y)

print("=" * 70)
print("ENGINE IGNITION SUCCESSFUL - FULL DIAGNOSTICS")
print("=" * 70)

# # 1. Dataset Profile
# print("\n1. DATASET PROFILE:")
# print(f"   Rows: {automl.report().n_rows}")
# print(f"   Columns: {automl.report().n_columns}")
# print(f"   Task Type: {automl.report().task_type}")
# print(f"   Hash: {automl.report().dataset_hash}")

# # 2. Risk Score
# print("\n2. DATASET RISK SCORE:")
# risk = automl.get_risk_score()
# print(f"   Score: {risk['score']}/100")
# print(f"   Category: {risk['category']}")
# if risk['factors']:
#     print("   Risk Factors:")
#     for factor_name, factor_desc in risk['factors'].items():
#         print(f"      • {factor_desc}")

# # 3. Feature Importance
# print("\n3. FEATURE IMPORTANCE (Top 5):")
# importance = automl.get_feature_importance()
# for i, (feat, imp) in enumerate(list(importance.items())[:5], 1):
#     print(f"   {i}. {feat}: {imp:.4f}")

# # 4. Preprocessing Suggestions
# print("\n4. PREPROCESSING SUGGESTIONS:")
# suggestions = automl.get_preprocessing_suggestions()
# for section, sug_list in suggestions.items():
#     section_name = section.replace('_', ' ').title()
#     print(f"   {section_name}:")
#     if isinstance(sug_list, list):
#         for s in sug_list[:2]:  # Show first 2
#             print(f"      • {s}")
#     else:
#         print(f"      • {sug_list}")

# print("\n" + "=" * 70)
# print("Generating comprehensive PDF report...")
# print("=" * 70)
pdf_file = automl.generate_report()

print(f"\n✓ Report saved: {pdf_file}")
# print("\nReport includes:")
# print("  • Risk Score Assessment")
# print("  • Data Quality Analysis")
# print("  • Feature Distributions")
# print("  • Correlation Heatmap")
# print("  • SHAP Feature Importance")
# print("  • Preprocessing Recommendations")
# print("  • Strategic Automation Suggestions")
# print("\n" + "=" * 70)
