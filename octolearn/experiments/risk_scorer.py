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
            points = 10
            score += points
            self.risk_factors["id_columns"] = {
                "score": points,
                "description": f"ID-like columns detected: {self.profile.id_like_columns}"
            }

        # =====================================================================
        # DATA LEAKAGE (25)
        # =====================================================================
        if hasattr(self.profile, "leakage_suspects") and self.profile.leakage_suspects:
            points = 25
            score += points
            self.risk_factors["leakage"] = {
                "score": points,
                "description": f"Potential leakage suspects: {self.profile.leakage_suspects}"
            }

        # =====================================================================
        # LOW VARIANCE (5)
        # =====================================================================
        if self.profile.low_variance_columns:
            points = 5
            score += points
            self.risk_factors["low_variance"] = {
                "score": points,
                "description": f"Low variance columns: {len(self.profile.low_variance_columns)}"
            }

        # =====================================================================
        # DUPLICATE ROWS (15)
        # =====================================================================
        dup_count = getattr(self.profile, "duplicate_rows", 0)
        if dup_count > 0:
            dup_pct = (dup_count / rows) * 100 if rows else 0
            if dup_pct > 10:
                points = 15
            elif dup_pct > 5:
                points = 10
            else:
                points = 5
            
            score += points
            self.risk_factors["duplicates"] = {
                "score": points,
                "description": f"{dup_count} duplicate rows ({dup_pct:.1f}%)"
            }

        # =====================================================================
        # CLASS IMBALANCE (15)
        # =====================================================================
        imbalance = getattr(self.profile, "imbalance_ratio", None)
        if imbalance is not None:
            points = 0
            if imbalance < 0.70:
                points = 15
                desc = f"Severe imbalance ratio: {imbalance:.3f}"
            elif imbalance < 0.85:
                points = 10
                desc = f"Moderate imbalance ratio: {imbalance:.3f}"
            elif imbalance < 0.95:
                points = 5
                desc = f"Minor imbalance ratio: {imbalance:.3f}"
            
            if points > 0:
                score += points
                self.risk_factors["imbalance"] = {
                    "score": points,
                    "description": desc
                }

        # =====================================================================
        # MISSING VALUES (20)
        # =====================================================================
        if self.profile.missing_ratio:
            avg_missing = sum(self.profile.missing_ratio.values()) / len(self.profile.missing_ratio)
            points = 0
            if avg_missing > 0.5:
                points = 20
            elif avg_missing > 0.3:
                points = 15
            elif avg_missing > 0.1:
                points = 10
            elif avg_missing > 0.05:
                points = 5
            
            if points > 0:
                score += points
                self.risk_factors["missing_values"] = {
                    "score": points,
                    "description": f"Average missing ratio: {avg_missing:.2f}"
                }

        # =====================================================================
        # CONSTANT COLUMNS (10)
        # =====================================================================
        if self.profile.constant_columns:
            points = 10
            score += points
            self.risk_factors["constant_columns"] = {
                "score": points,
                "description": f"Constant columns: {self.profile.constant_columns}"
            }

        # =====================================================================
        # HIGH CARDINALITY (10)
        # =====================================================================
        if hasattr(self.profile, "high_cardinality_cols") and self.profile.high_cardinality_cols:
            points = 10
            score += points
            self.risk_factors["high_cardinality"] = {
                "score": points,
                "description": f"High cardinality columns: {len(self.profile.high_cardinality_cols)}"
            }

        # =====================================================================
        # FEATURE TO SAMPLE RATIO (10)
        # =====================================================================
        ratio = cols / rows if rows else 0
        points = 0
        if ratio > 0.5:
            points = 10
        elif ratio > 0.1:
            points = 5
        
        if points > 0:
            score += points
            self.risk_factors["feature_ratio"] = {
                "score": points,
                "description": f"High feature/sample ratio: {ratio:.2f}"
            }

        # =====================================================================
        # SAMPLE SIZE (5)
        # =====================================================================
        points = 0
        desc = ""
        if rows < 50:
            points = 5
            desc = f"Very small dataset: {rows} rows"
        elif rows < 100:
            # Note: Logic in original was ambiguous, assuming it adds 0 points but warns?
            # Or maybe it was meant to add points. Assuming low risk for now.
            desc = f"Small dataset: {rows} rows"
        
        if points > 0 or desc:
            if points > 0: score += points
            self.risk_factors["small_sample"] = {
                "score": points,
                "description": desc
            }

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