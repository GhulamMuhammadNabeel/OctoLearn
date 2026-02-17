import pandas as pd


class RiskScorer:
    """
    Comprehensive data quality risk scoring (0-100) for datasets.
    """

    def __init__(self, profile, X: pd.DataFrame):
        self.profile = profile
        self.X = X
        self.risk_factors = {}

    # --------------------------------------------------------
    # Main Risk Score
    # --------------------------------------------------------
    def calculate_risk_score(self):
        score = 0
        self.risk_factors = {}

        rows, cols = self.profile.shape

        # =====================================================================
        # ID-LIKE COLUMNS (10)
        # =====================================================================
        if self.profile.id_like_columns:
            score += 10
            self.risk_factors["id_columns"] = (
                f"ID-like columns detected: {self.profile.id_like_columns}"
            )

        # =====================================================================
        # DATA LEAKAGE (25)
        # =====================================================================
        if hasattr(self.profile, "leakage_suspects") and self.profile.leakage_suspects:
            score += 25
            self.risk_factors["leakage"] = (
                f"Potential leakage suspects: {self.profile.leakage_suspects}"
            )

        # =====================================================================
        # LOW VARIANCE (5)
        # =====================================================================
        if self.profile.low_variance_columns:
            score += 5
            self.risk_factors["low_variance"] = (
                f"Low variance columns: {len(self.profile.low_variance_columns)}"
            )

        # =====================================================================
        # DUPLICATE ROWS (15)
        # =====================================================================
        dup_count = getattr(self.profile, "duplicate_rows", 0)
        if dup_count > 0:
            dup_pct = (dup_count / rows) * 100 if rows else 0
            if dup_pct > 10:
                score += 15
            elif dup_pct > 5:
                score += 10
            else:
                score += 5
            self.risk_factors["duplicates"] = f"{dup_count} duplicate rows ({dup_pct:.1f}%)"

        # =====================================================================
        # CLASS IMBALANCE (15)
        # =====================================================================
        imbalance = getattr(self.profile, "imbalance_ratio", None)
        if imbalance is not None:
            if imbalance < 0.70:
                score += 15
                self.risk_factors["imbalance"] = f"Severe imbalance ratio: {imbalance:.3f}"
            elif imbalance < 0.85:
                score += 10
                self.risk_factors["imbalance"] = f"Moderate imbalance ratio: {imbalance:.3f}"
            elif imbalance < 0.95:
                score += 5
                self.risk_factors["imbalance"] = f"Minor imbalance ratio: {imbalance:.3f}"

        # =====================================================================
        # MISSING VALUES (20)
        # =====================================================================
        if self.profile.missing_ratio:
            avg_missing = sum(self.profile.missing_ratio.values()) / len(self.profile.missing_ratio)
            if avg_missing > 0.5:
                score += 20
            elif avg_missing > 0.3:
                score += 15
            elif avg_missing > 0.1:
                score += 10
            elif avg_missing > 0.05:
                score += 5
            self.risk_factors["missing_values"] = f"Average missing ratio: {avg_missing:.2f}"

        # =====================================================================
        # CONSTANT COLUMNS (10)
        # =====================================================================
        if self.profile.constant_columns:
            score += 10
            self.risk_factors["constant_columns"] = f"Constant columns: {self.profile.constant_columns}"

        # =====================================================================
        # HIGH CARDINALITY (10)
        # =====================================================================
        if hasattr(self.profile, "high_cardinality_cols") and self.profile.high_cardinality_cols:
            score += 10
            self.risk_factors["high_cardinality"] = (
                f"High cardinality columns: {len(self.profile.high_cardinality_cols)}"
            )

        # =====================================================================
        # FEATURE TO SAMPLE RATIO (10)
        # =====================================================================
        ratio = cols / rows if rows else 0
        if ratio > 0.5:
            score += 10
            self.risk_factors["feature_ratio"] = f"High feature/sample ratio: {ratio:.2f}"
        elif ratio > 0.1:
            score += 5

        # =====================================================================
        # SAMPLE SIZE (5)
        # =====================================================================
        if rows < 50:
            score += 5
            self.risk_factors["small_sample"] = f"Very small dataset: {rows} rows"
        elif rows < 100:
            self.risk_factors["small_sample"] = f"Small dataset: {rows} rows"

        # Cap score
        score = min(score, 100)

        # =====================================================================
        # CATEGORY
        # =====================================================================
        if score <= 30:
            category = "Low Risk"
        elif score <= 60:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        return score, category, self.risk_factors
