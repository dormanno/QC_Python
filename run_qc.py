"""Command-line interface for running QC orchestration."""

import logging
from ColumnNames import pnl_column
from FeatureNormalizer import FeatureNormalizer
from QCEngine import QCEngine
from QC_Orchestrator import QCOrchestrator


def main():
    """Main entry point for QC orchestration."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define QC features
    qc_features = [
        pnl_column.START, 
        *pnl_column.SLICE_COLUMNS, 
        pnl_column.TOTAL, 
        pnl_column.EXPLAINED, 
        pnl_column.UNEXPLAINED
    ]
    
    # Define aggregator weights
    weight_if = 0.4
    weight_rz = 0.3
    weight_roll = 0.2
    weight_iqr = 0.1
    roll_window = 20
    
    # Create feature normalizer
    normalizer = FeatureNormalizer(features=qc_features)
    
    # Create QC Engine
    qc_engine = QCEngine(
        qc_features=qc_features,
        weight_if=weight_if,
        weight_rz=weight_rz,
        weight_roll=weight_roll,
        weight_iqr=weight_iqr,
        roll_window=roll_window
    )
    
    # Get input path from user
    path = input("Enter full path to PnL_Input.csv: ").strip()
    
    try:
        orchestrator = QCOrchestrator(
            normalizer=normalizer,
            qc_engine=qc_engine
        )
        out_path = orchestrator.run(path)
        print(f"\n=== QC Processing Complete ===")
        print(f"Output file: {out_path}")
    except Exception as e:
        logging.getLogger(__name__).exception("QC orchestration failed")
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
