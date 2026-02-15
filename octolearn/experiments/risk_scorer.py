class RiskScorer:
    """Comprehensive data quality risk scoring (0-100)."""

    def __init__(self, profile, X):
        """
        Initialize RiskScorer.
        
        Parameters
        ----------
        profile : DatasetProfile
            Dataset profile from DataProfiler
        X : pd.DataFrame
            Feature dataframe
        """
        self.profile = profile
        self.X = X
        self.risk_factors = {}

    def calculate_risk_score(self):
        """
        Calculate comprehensive risk score (0-100).
        
        Returns
        -------
        score : int
            Risk score (0-100)
        category : str
            Risk category (Low/Moderate/High)
        factors : dict
            Detailed risk factors
        """
        score = 0
        self.risk_factors = {}
        
        # =====================================================================
        # ID-LIKE COLUMNS (Risk: 10 points)
        # =====================================================================
        if hasattr(self.profile, 'id_like_columns') and self.profile.id_like_columns:
            score += 10
            self.risk_factors["id_columns"] = f"ID-like columns detected: {self.profile.id_like_columns}"
        
        # =====================================================================
        # POTENTIAL DATA LEAKAGE (Risk: 25 points)
        # =====================================================================
        if hasattr(self.profile, 'leakage_suspects') and self.profile.leakage_suspects:
            score += 25
            self.risk_factors["leakage"] = f"Potential leakage suspects: {self.profile.leakage_suspects}"
        
        # =====================================================================
        # LOW VARIANCE COLUMNS (Risk: 5 points)
        # =====================================================================
        if hasattr(self.profile, 'low_variance_columns') and self.profile.low_variance_columns:
            score += 5
            self.risk_factors["low_variance"] = f"Low variance columns: {len(self.profile.low_variance_columns)}"
        
        # =====================================================================
        # DUPLICATE ROWS (Risk: 15 points)
        # =====================================================================
        if hasattr(self.profile, 'duplicate_rows') and self.profile.duplicate_rows > 0:
            dup_pct = (self.profile.duplicate_rows / self.profile.n_rows) * 100
            if dup_pct > 10:
                score += 15
            elif dup_pct > 5:
                score += 10
            else:
                score += 5
            self.risk_factors["duplicates"] = f"{self.profile.duplicate_rows} duplicate rows ({dup_pct:.1f}%)"
        
        # =====================================================================
        # CLASS IMBALANCE (Risk: 15 points)
        # =====================================================================
        if hasattr(self.profile, 'imbalance_ratio') and self.profile.imbalance_ratio:
            if self.profile.imbalance_ratio > 0.95:
                score += 15
                self.risk_factors["imbalance"] = f"Severe imbalance ratio: {self.profile.imbalance_ratio:.3f}"
            elif self.profile.imbalance_ratio > 0.85:
                score += 10
                self.risk_factors["imbalance"] = f"Moderate imbalance ratio: {self.profile.imbalance_ratio:.3f}"
            elif self.profile.imbalance_ratio > 0.70:
                score += 5
                self.risk_factors["imbalance"] = f"Minor imbalance ratio: {self.profile.imbalance_ratio:.3f}"
        
        # =====================================================================
        # MISSING VALUES (Risk: 20 points)
        # =====================================================================
        if hasattr(self.profile, 'missing_report') and self.profile.missing_report:
            total_missing = sum(self.profile.missing_report.values())
            avg_missing = total_missing / len(self.profile.missing_report) if self.profile.missing_report else 0
            
            if avg_missing > 50:
                score += 20
            elif avg_missing > 30:
                score += 15
            elif avg_missing > 10:
                score += 10
            elif avg_missing > 5:
                score += 5
            
            if total_missing > 0:
                self.risk_factors["missing_values"] = f"Average missing percentage: {avg_missing:.1f}%"
        
        # =====================================================================
        # CONSTANT COLUMNS (Risk: 10 points)
        # =====================================================================
        if hasattr(self.profile, 'constant_columns') and self.profile.constant_columns:
            score += 10
            self.risk_factors["constant_columns"] = f"Constant/single-value columns: {self.profile.constant_columns}"
        
        # =====================================================================
        # HIGH CARDINALITY (Risk: 10 points)
        # =====================================================================
        if hasattr(self.profile, 'high_cardinality_cols') and self.profile.high_cardinality_cols:
            score += 10
            self.risk_factors["high_cardinality"] = f"High cardinality features: {len(self.profile.high_cardinality_cols)}"
        
        # =====================================================================
        # FEATURE-TO-SAMPLE RATIO (Risk: 10 points)
        # =====================================================================
        if hasattr(self.profile, 'n_rows') and hasattr(self.profile, 'n_columns'):
            ratio = self.profile.n_columns / self.profile.n_rows if self.profile.n_rows > 0 else 0
            if ratio > 0.5:  # More features than rows
                score += 10
                self.risk_factors["feature_ratio"] = f"High feature-to-sample ratio: {ratio:.2f}"
            elif ratio > 0.1:  # Many features
                score += 5
        
        # =====================================================================
        # SAMPLE SIZE (Risk: 5 points)
        # =====================================================================
        if hasattr(self.profile, 'n_rows'):
            if self.profile.n_rows < 50:
                score += 5
                self.risk_factors["small_sample"] = f"Very small dataset: {self.profile.n_rows} rows"
            elif self.profile.n_rows < 100:
                self.risk_factors["small_sample"] = f"Small dataset: {self.profile.n_rows} rows"
        
        # Cap score at 100
        score = min(score, 100)
        
        # =====================================================================
        # DETERMINE RISK CATEGORY
        # =====================================================================
        if score <= 30:
            category = "Low Risk"
        elif score <= 60:
            category = "Moderate Risk"
        else:
            category = "High Risk"
        
        return score, category, self.risk_factors
