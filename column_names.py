from dataclasses import dataclass


@dataclass(frozen=True)
class MainColumn:
    """Main trade identifier columns"""
    TRADE: str = "TradeID"
    BOOK: str = "Book"
    TRADE_TYPE: str = "TradeType"
    DATE: str = "Date"


@dataclass(frozen=True)
class PnLColumn:
    """PnL-related columns"""
    START: str = "Start_PV"
    END: str = "End_PV"
    TOTAL: str = "Total_PnL"
    EXPLAINED: str = "Explained"
    UNEXPLAINED: str = "Residual"
    TOTAL_JUMP: str = "TotalJump"
    UNEXPLAINED_JUMP: str = "ResidualJump"
    BASIS_COF: str = "Basis_CoF_PnL"
    RECOVERY_RATE: str = "Recovery_Rate_PnL"
    ROLL: str = "Roll_PnL"
    RATES: str = "Rates_PnL"
    MISC: str = "Misc_PnL"
    MODEL: str = "Model_PnL"
    MODS: str = "Mods_PnL"
    CREDIT_INDEX: str = "Credit_Index_PnL"
    INDEX_CORRELATION: str = "IndexCorrelation_PnL"
    CREDIT_SINGLE: str = "Credit_Single_PnL"
    
    # Derived lists (initialized in __post_init__)
    SLICE_COLUMNS: list = None
    INPUT_FEATURES: list = None
    QC_FEATURES: list = None
    
    def __post_init__(self):
        """Initialize derived column lists after instantiation."""
        object.__setattr__(self, 'SLICE_COLUMNS', [
            self.BASIS_COF, self.RECOVERY_RATE, self.ROLL, 
            self.RATES, self.MISC, self.MODEL, self.MODS, 
            self.CREDIT_INDEX, self.INDEX_CORRELATION, self.CREDIT_SINGLE
        ])
        object.__setattr__(self, 'INPUT_FEATURES', [
            self.START, *self.SLICE_COLUMNS, self.END
        ])
        object.__setattr__(self, 'QC_FEATURES', [
            self.START, *self.SLICE_COLUMNS, self.TOTAL, 
            self.EXPLAINED, self.UNEXPLAINED
        ])


@dataclass(frozen=True)
class QCColumn:
    """QC output columns"""
    QC_FLAG: str = "QC_Flag"
    IF_SCORE: str = "IF_score"
    ROBUST_Z_SCORE: str = "RobustZ_score"
    ROLLING_SCORE: str = "Rolling_score"
    IQR_SCORE: str = "IQR_score"
    AGGREGATED_SCORE: str = "QC_Aggregated"
    
    # Derived lists (initialized in __post_init__)
    SCORE_COLUMNS: list = None
    
    def __post_init__(self):
        """Initialize derived score columns list after instantiation."""
        object.__setattr__(self, 'SCORE_COLUMNS', [
            self.IF_SCORE, self.ROBUST_Z_SCORE, self.ROLLING_SCORE, 
            self.IQR_SCORE, self.AGGREGATED_SCORE, self.QC_FLAG
        ])


# Create singleton instances for easier access
main_column = MainColumn()
pnl_column = PnLColumn()
qc_column = QCColumn()