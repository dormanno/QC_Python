from dataclasses import dataclass
from abc import ABC


@dataclass(frozen=True)
class FeatureColumnSet(ABC):
    """Abstract base class for columns with INPUT_FEATURES and QC_FEATURES"""
    INPUT_FEATURES: list = None
    QC_FEATURES: list = None
    ENGINEERED_FEATURES: list = None


@dataclass(frozen=True)
class MainColumnSet:
    """Main trade identifier columns"""
    TRADE: str = "TradeID"
    BOOK: str = "Book"
    TRADE_TYPE: str = "TradeType"
    DATE: str = "Date"
    MAIN_COLUMNS: list = None

    def __post_init__(self):
        object.__setattr__(self, 'MAIN_COLUMNS', [
                self.TRADE, 
                self.BOOK,
                self.TRADE_TYPE,
                self.DATE
            ])

@dataclass(frozen=True)
class CreditDeltaSingleColumnSet(FeatureColumnSet):
    """Credit Delta Single related columns"""
    CREDIT_DELTA_SINGLE: str = "CreditDeltaSingle"

    def __post_init__(self):
        object.__setattr__(self, 'QC_FEATURES', [
                self.CREDIT_DELTA_SINGLE
            ])
        object.__setattr__(self, 'INPUT_FEATURES', [
                self.CREDIT_DELTA_SINGLE
            ])
        object.__setattr__(self, 'ENGINEERED_FEATURES', [])# No engineered features for CreditDeltaSingle

@dataclass(frozen=True)
class CreditDeltaIndexColumnSet(FeatureColumnSet):
    """Credit Delta Index related columns"""
    CREDIT_DELTA_INDEX: str = "CreditDeltaIndex"

    def __post_init__(self):
        object.__setattr__(self, 'QC_FEATURES', [
                self.CREDIT_DELTA_INDEX
            ])
        object.__setattr__(self, 'INPUT_FEATURES', [
                self.CREDIT_DELTA_INDEX
            ])
        object.__setattr__(self, 'ENGINEERED_FEATURES', [])# No engineered features for CreditDeltaIndex

@dataclass(frozen=True)
class PnLColumnSet(FeatureColumnSet):
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
        object.__setattr__(self, 'ENGINEERED_FEATURES', [
            self.TOTAL, self.EXPLAINED, self.UNEXPLAINED,
            self.TOTAL_JUMP, self.UNEXPLAINED_JUMP
        ])


@dataclass(frozen=True)
class QCColumnSet:
    """QC output columns"""
    QC_FLAG: str = "QC_Flag"
    IF_SCORE: str = "IF_score"
    ROBUST_Z_SCORE: str = "RobustZ_score"
    ROLLING_SCORE: str = "Rolling_score"
    IQR_SCORE: str = "IQR_score"
    AGGREGATED_SCORE: str = "QC_Aggregated"
    LOF_SCORE: str = "LOF_score"
    ECDF_SCORE: str = "ECDF_score"
    
    # Derived lists (initialized in __post_init__)
    SCORE_COLUMNS: list = None
    
    def __post_init__(self):
        """Initialize derived score columns list after instantiation."""
        object.__setattr__(self, 'SCORE_COLUMNS', [
            self.IF_SCORE, 
            self.ROBUST_Z_SCORE, 
            self.ROLLING_SCORE, 
            self.IQR_SCORE, 
            self.LOF_SCORE,
            self.ECDF_SCORE,
            self.AGGREGATED_SCORE,             
            self.QC_FLAG
        ])


# Create singleton instances for easier access
main_column = MainColumnSet()
pnl_column = PnLColumnSet()
qc_column = QCColumnSet()
cds_column = CreditDeltaSingleColumnSet()
cdi_column = CreditDeltaIndexColumnSet()