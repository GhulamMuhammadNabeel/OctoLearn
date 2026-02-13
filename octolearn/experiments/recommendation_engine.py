class RecommendationEngine:

    def __init__(self, profile):
        self.profile = profile

    def generate(self):
        recommendations = []

        if self.profile.imbalance_ratio and self.profile.imbalance_ratio > 0.8:
            recommendations.append(
                "Severe class imbalance detected. Consider SMOTE or class_weight balancing."
            )

        if len(self.profile.skewed_columns) > 0:
            recommendations.append(
                "Highly skewed features detected. Log transformation may improve performance."
            )

        if len(self.profile.constant_columns) > 0:
            recommendations.append(
                "Constant columns found. Remove them to reduce noise."
            )

        if len(self.profile.high_cardinality_cols) > 0:
            recommendations.append(
                "High cardinality categorical features detected. Prefer target encoding."
            )

        if self.profile.duplicate_rows > 0:
            recommendations.append(
                f"Duplicate rows exist ({self.profile.duplicate_rows}). Consider deduplication."
            )

        if not recommendations:
            recommendations.append("Dataset looks structurally healthy.")

        return recommendations
