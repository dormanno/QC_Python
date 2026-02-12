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
- **Date Range:** [01.01.2025] to [30.06.2025]
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
- **Rows:** [Number of rows]
- **Columns:** [Number of columns]
- **Date Range:** [Start date] to [End date]
- **Key Features:**
  - [Feature 1 and its description]
  - [Feature 2 and its description]
  - [etc.]

**Known Data Patterns:**
- [Describe typical patterns]

**Known Outliers/Edge Cases:**
- [Describe any known outliers]

**Missing Data:**
- [Describe any missing data]

---

### 3. Credit Delta Index (CDI) Data
**File Name:** [Add file name here]

**Description:**
- [Describe the CDI data]

**Dataset Characteristics:**
- **Rows:** [Number of rows]
- **Columns:** [Number of columns]
- **Date Range:** [Start date] to [End date]
- **Key Features:**
  - [Feature 1 and its description]
  - [Feature 2 and its description]
  - [etc.]

**Known Data Patterns:**
- [Describe typical patterns]

**Known Outliers/Edge Cases:**
- [Describe any known outliers]

**Missing Data:**
- [Describe any missing data]

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
- [What outliers should be detected?]
- [What should pass QC?]
- [Any specific edge cases to watch for?]

**For CDS Data:**
- [Expected detection patterns]

**For CDI Data:**
- [Expected detection patterns]

---

## Data Quality Issues to Test

### High Priority Issues
1. [Issue type 1 - e.g., extreme spikes in feature X]
2. [Issue type 2 - e.g., gradual drift in feature Y]
3. [Issue type 3 - e.g., missing data patterns]

### Medium Priority Issues
1. [Issue type]
2. [Issue type]

### Edge Cases
1. [Edge case 1]
2. [Edge case 2]

---

## Notes for AI Assistant

**Context for Better Suggestions:**
- [Add any additional context about your data]
- [Describe the business context]
- [Mention any domain-specific considerations]
- [Note any regulatory or compliance requirements]

**Common Issues Encountered:**
- [Issue 1 and how it manifests]
- [Issue 2 and how it manifests]

**Testing Goals:**
- [What you're trying to validate]
- [What success looks like]

---

## Update History
- **[Date]:** Initial documentation created
- **[Date]:** [Description of update]
