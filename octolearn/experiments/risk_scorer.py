"""
Risk scoring engine - calculates data quality score (0-100)
"""


class RiskScorer:

    def __init__(self, profile, X):
        self.profile = profile
        self.X = X
        self.risk_factors = {}

    def calculate_risk_score(self) -> tuple:
        """
        Calculate dataset risk score (0-100) where:
        - 0-30: High quality
        - 31-60: Moderate quality
        - 61-100: High risk
        
        Returns: (score, category, risk_factors_dict)
        """
        score = 0
        
        # Missing data penalty (max 20 points)
        max_missing = max(self.profile.missing_report.values()) if self.profile.missing_report else 0
        if max_missing > 50:
            score += 20
            self.risk_factors['high_missing'] = f"Max {max_missing}% missing data"
        elif max_missing > 20:
            score += 15
            self.risk_factors['moderate_missing'] = f"Max {max_missing}% missing data"
        elif max_missing > 0:
            score += 5
            self.risk_factors['low_missing'] = f"Max {max_missing}% missing data"

        # Duplicate rows penalty (max 15 points)
        dup_ratio = self.profile.duplicate_rows / self.profile.n_rows * 100
        if dup_ratio > 10:
            score += 15
            self.risk_factors['high_duplicates'] = f"{dup_ratio:.1f}% duplicate rows"
        elif dup_ratio > 2:
            score += 10
            self.risk_factors['moderate_duplicates'] = f"{dup_ratio:.1f}% duplicate rows"
        elif dup_ratio > 0:
            score += 3
            self.risk_factors['low_duplicates'] = f"{dup_ratio:.1f}% duplicate rows"

        # Class imbalance penalty (max 15 points)
        if self.profile.imbalance_ratio:
            if self.profile.imbalance_ratio > 0.95:
                score += 15
                self.risk_factors['severe_imbalance'] = f"{self.profile.imbalance_ratio:.2%} majority class"
            elif self.profile.imbalance_ratio > 0.8:
                score += 10
                self.risk_factors['high_imbalance'] = f"{self.profile.imbalance_ratio:.2%} majority class"
            elif self.profile.imbalance_ratio > 0.6:
                score += 5
                self.risk_factors['moderate_imbalance'] = f"{self.profile.imbalance_ratio:.2%} majority class"

        # Skewed features penalty (max 10 points)
        if len(self.profile.skewed_columns) > self.profile.n_columns * 0.5:
            score += 10
            self.risk_factors['high_skew'] = f"{len(self.profile.skewed_columns)} highly skewed features"
        elif len(self.profile.skewed_columns) > 0:
            score += 5
            self.risk_factors['moderate_skew'] = f"{len(self.profile.skewed_columns)} skewed features"

        # Constant columns penalty (max 10 points)
        if len(self.profile.constant_columns) > 0:
            score += 10
            self.risk_factors['constant_cols'] = f"{len(self.profile.constant_columns)} constant columns"

        # High cardinality penalty (max 10 points)
        if len(self.profile.high_cardinality_cols) > self.profile.n_columns * 0.3:
            score += 10
            self.risk_factors['high_cardinality'] = f"{len(self.profile.high_cardinality_cols)} high-cardinality features"
        elif len(self.profile.high_cardinality_cols) > 0:
            score += 5
            self.risk_factors['moderate_cardinality'] = f"{len(self.profile.high_cardinality_cols)} high-cardinality features"

        # Feature-to-sample ratio penalty (max 10 points)
        feature_sample_ratio = self.profile.n_columns / self.profile.n_rows
        if feature_sample_ratio > 0.1:
            score += 10
            self.risk_factors['high_ratio'] = f"Features/samples ratio: {feature_sample_ratio:.2f}"
        elif feature_sample_ratio > 0.01:
            score += 5
            self.risk_factors['moderate_ratio'] = f"Features/samples ratio: {feature_sample_ratio:.2f}"

        # Small sample size penalty (max 5 points)
        if self.profile.n_rows < 100:
            score += 5
            self.risk_factors['small_dataset'] = f"Only {self.profile.n_rows} samples"
        elif self.profile.n_rows < 500:
            score += 2
            self.risk_factors['moderate_size'] = f"Only {self.profile.n_rows} samples"

        # Cap at 100
        score = min(score, 100)

        # Determine category
        if score <= 30:
            category = "Low Risk"
        elif score <= 60:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        return score, category, self.risk_factors
