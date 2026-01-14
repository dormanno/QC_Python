# identifiers for column names used in QC analysis
TRADE = "TradeID"
BOOK = "Book"
TRADE_TYPE = "TradeType"
# temporal column name
DATE = "Date"
# PnL features
START = "Start_PV"
END = "End_PV"

TOTAL = "Total_PnL"
EXPLAINED = "Explained"
UNEXPLAINED = "Residual"
TOTAL_JUMP = "TotalJump"
UNEXPLAINED_JUMP = "ResidualJump"
PNL_SLICES = [
    "Basis_CoF_PnL", "Recovery_Rate_PnL", "Roll_PnL", "Rates_PnL", "Misc_PnL", "Model_PnL", "Mods_PnL", "Credit_Index_PnL", "IndexCorrelation_PnL", "Credit_Single_PnL"
]
PNL_INPUT_FEATURES = [START, *PNL_SLICES, END]
PNL_FEATURES = [START, *PNL_SLICES, TOTAL, EXPLAINED, UNEXPLAINED]
# QC output columns
QC_FLAG = "QC_Flag"
IF_SCORE = "IF_score"
ROBUST_Z_SCORE = "RobustZ_score"
ROLLING_SCORE = "Rolling_score"
IQR_SCORE = "IQR_score"
AGGREGATED_SCORE = "QC_Aggregated"
SCORE_FLAG = "QC_Flag"
DEFAULT_SCORES = [IF_SCORE, ROBUST_Z_SCORE, ROLLING_SCORE, IQR_SCORE, AGGREGATED_SCORE, SCORE_FLAG]