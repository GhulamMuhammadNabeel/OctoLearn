class RecommendationEngine:

    def __init__(self, profile):
        self.profile = profile

    def generate(self):
        recs = []

        if self.profile.id_like_columns:
            recs.append(f"Remove identifier columns: {self.profile.id_like_columns}")

        if self.profile.leakage_suspects:
            recs.append(f"Investigate target leakage in: {self.profile.leakage_suspects}")

        if self.profile.low_variance_columns:
            recs.append(f"Drop low variance columns: {self.profile.low_variance_columns}")

        if self.profile.imbalance_ratio and self.profile.imbalance_ratio > 0.8:
            recs.append("Apply class balancing techniques.")

        if not recs:
            recs.append("Dataset structurally sound. Proceed to modeling.")

        return recs
