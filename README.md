# QC_Python – Automated Quality Control for Risk Calculations

This project implements an **automated quality control (QC) framework** for daily risk calculation results in investment banking.  
It applies several complementary outlier-detection methods to PnL measures and Present Values, aiming to flag potential data errors (e.g., misquoted curves).

## Features

- **Isolation Forest QC** – anomaly detection using ensemble trees
- **Robust Z-Score QC** – deviation from per-trade robust statistics (median / MAD)
- **IQR QC** – Tukey’s interquartile rule for detecting outliers
- **Rolling Z QC** – rolling-window standardization with per-trade buffers
- **Aggregation** – weighted daily score combining all methods

The system uses a **walk-forward evaluation**:
1. Split dataset into **training window** (first 60 days) and **out-of-sample (OOS)** window (rest of history).
2. Fit QC methods on the training set.
3. Iterate through OOS days and compute scores per trade, per method.
4. Aggregate into one daily QC score per trade.

## Usage

python run_qc.py

You will be prompted to enter the path to your input CSV (containing risk measures and PnL slices).

The script outputs per-trade QC scores for each out-of-sample day and prints a sample of the most anomalous trades.

## Notes
Input data must include trade identifiers, dates, Present Values, and PnL slices as specified in ColumnNames.py.
Training period length (TRAIN_DAYS) and method parameters can be tuned in QC_Orchestrator.py.

## Acknowledgements

This project was developed with assistance from ChatGPT (OpenAI, GPT-5 model), which helped structure the code, design the walk-forward methodology, and implement the QC methods in an object-oriented style.
