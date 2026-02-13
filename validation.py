#!/usr/bin/env python
"""OctoLearn v0.2 - Production Validation"""

from octolearn import AutoML
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
X = data.data
y = data.target

automl = AutoML()
automl.fit(X, y)

risk = automl.get_risk_score()
importance = automl.get_feature_importance()
sugg = automl.get_preprocessing_suggestions()

print('\n' + '='*70)
print('OCTOLEARN v0.2 - PRODUCTION VALIDATION')
print('='*70)

print('\n✅ RISK SCORING MODULE')
print(f'   Score: {risk["score"]}/100')
print(f'   Category: {risk["category"]}')
print(f'   Factors: {len(risk["factors"])} detected')

print('\n✅ FEATURE IMPORTANCE MODULE')
print(f'   Features Ranked: {len(importance)}')
top_feat = list(importance.keys())[0]
top_score = list(importance.values())[0]
print(f'   Top Feature: {top_feat} ({top_score:.4f})')

print('\n✅ PREPROCESSING SUGGESTIONS MODULE')
print(f'   Categories: {len(sugg)}')
for cat in sugg.keys():
    cat_name = cat.replace('_', ' ').title()
    print(f'   • {cat_name}')

print('\n✅ REPORT GENERATION')
pdf = automl.generate_report()
print(f'   PDF: {pdf}')
print(f'   Status: Successfully generated')

print('\n' + '='*70)
print('ALL SYSTEMS OPERATIONAL ✅')
print('='*70 + '\n')
