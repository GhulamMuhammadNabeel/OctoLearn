class RiskScorer:

    def __init__(self, profile, X):
        self.profile = profile
        self.X = X
        self.risk_factors = {}

    def calculate_risk_score(self):

        score = 0

        if self.profile.id_like_columns:
            score += 10
            self.risk_factors["id_columns"] = f"ID-like columns detected: {self.profile.id_like_columns}"

        if self.profile.leakage_suspects:
            score += 25
            self.risk_factors["leakage"] = f"Potential leakage: {self.profile.leakage_suspects}"

        if self.profile.low_variance_columns:
            score += 5

        if self.profile.duplicate_rows > 0:
            score += 5

        if self.profile.imbalance_ratio and self.profile.imbalance_ratio > 0.85:
            score += 10

        score = min(score, 100)

        if score <= 30:
            category = "Low Risk"
        elif score <= 60:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        return score, category, self.risk_factors
