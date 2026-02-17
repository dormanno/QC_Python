# Test Data Documentation

## Overview
This document describes the test datasets used for QC (Quality Control) validation and testing.

**Data Location:** `C:\Users\dorma\Documents\UEK_Backup\Test`

---

## Datasets

### 1. PnL (Profit and Loss) Data
**File Name:** PnL_Input_Injected.csv

**Description:**
- file contains Risk Valuation results for a set of trades from an investment Bank. Trades get valuated every day. One row represents values calculated for a trade on a given date

**Dataset Characteristics:**
- **Rows:** 21620
- **Columns:** 17
- **Date Range:** [01.01.2025] to [30.06.2025] (129 days with observations)
- **Identification Columns:**
  - RecordType: distinguishes type of the row, if it is part of train data, OOS or one of injected artifial outliers
  - Book - book name, not used in assessment. Used only for reporting purpuses
  - TradeID - ID of a trade which has values calculated every day
  - TradeType - high level trade type, used to split whole dataset into parts, as values for different trade types behave differently
- **Temporal Column:**
  - Date - date of valuation
- **Feature Columns:**
  - Start_PV	
  - Basis_CoF_PnL	
  - Recovery_Rate_PnL	
  - Roll_PnL	
  - Rates_PnL
  - Misc_PnL
  - Model_PnL
  - Mods_PnL
  - Credit_Index_PnL
  - Credit_Single_PnL
  - IndexCorrelation_PnL
  - End_PV    

**Known Data Patterns:**
- First 3/4 of rows are used as train data, rest - OOS

**Known Outliers/Edge Cases:**
- 20-23 of June is a Roll date so values may naturally spike
- some trades are booked within period so they appear initially with mostly zero values at the first day
- some trades mature within period and disappear from the data set

**Missing Data:**
- as mentioned above if a trade is newly booked it has some values missed on the first day
- Identification and Temporal columns are always defined

---

### 2. Credit Delta Single (CDS) Data
**File Name:** CreditDeltaSingle_Input

**Description:**
- file contains Risk Valuation results for a set of trades from an investment Bank. Trades get valuated every day. One row represents values calculated for a trade on a given date

**Dataset Characteristics:**
- **Rows:** 21356
- **Columns:** 6
- **Date Range:** [01.01.2025] to [30.06.2025] (129 days with observations)
- **Identification Columns:**
  - RecordType: distinguishes type of the row, if it is part of train data, OOS or one of injected artifial outliers
  - Book - book name, not used in assessment. Used only for reporting purpuses
  - TradeID - ID of a trade which has values calculated every day
  - TradeType - high level trade type, used to split whole dataset into parts, as values for different trade types behave differently. 5 TradeTypes present within dataset: Basis (Bond Basis CDS), Basket (Basket CDS), Index (Index CDS), SingleName (classic vanilla CDS) and Tranche (Index Tranche CDS)
- **Temporal Column:**
  - Date - date of valuation
- **Feature Columns:**
  - CreditDeltaSingle - CreditDelta measure for parallel shift of the underlying curve

**Known Data Patterns:**
- First 3/4 of rows are used as train data, rest - OOS

**Known Outliers/Edge Cases:**
- 20-23 of June is a Roll date so values may naturally spike
- some trades are booked within period so they appear in the middle of the history
- some trades mature within period and disappear from the data set

**Missing Data:**
- No missing data is expected

---

### 3. Credit Delta Index (CDI) Data
- **Rows:** 1270
- **Columns:** 6
- **Date Range:** [01.01.2025] to [30.06.2025] (129 days with observations)
- **Identification Columns:**
  - RecordType: distinguishes type of the row, if it is part of train data, OOS or one of injected artifial outliers
  - Book - book name, not used in assessment. Used only for reporting purpuses
  - TradeID - ID of a trade which has values calculated every day
  - TradeType - high level trade type, used to split whole dataset into parts, as values for different trade types behave differently. Only 2 trade types are observed within dataset: Basis (Bond Basis CDS) and Index (Index CDS)
- **Temporal Column:**
  - Date - date of valuation
- **Feature Columns:**
  - CreditDeltaIndex - CreditDelta measure for parallel shift of IndexCurve

**Known Data Patterns:**
- First 3/4 of rows are used as train data, rest - OOS

**Known Outliers/Edge Cases:**
- 20-23 of June is a Roll date so values may naturally spike
- some trades are booked within period so they appear in the middle of the history
- some trades mature within period and disappear from the data set

**Missing Data:**
- No missing data is expected

---

## QC Method Testing Expectations

### Current Method Configuration
- **Isolation Forest:** 0.2 weight
- **Robust Z:** 0.1 weight
- **Rolling:** 0.1 weight (window: 20)
- **IQR:** 0.1 weight
- **LOF:** 0.2 weight
- **ECDF:** 0.2 weight
- **Hampel:** 0.1 weight

### Expected Behavior
**For PnL Data:**
- outliers injected within the framework by oulier injector classes
- most of unmodified records, few natural outliers are expected

---