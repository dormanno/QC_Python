"""Command-line interface for running QC orchestration."""

import logging
from column_names import pnl_column, qc_column
from Engine.feature_normalizer import FeatureNormalizer
from Engine.qc_engine import QCEngine
from qc_orchestrator import QCOrchestrator
from input import PnLInput


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
    weights = {
        qc_column.IF_SCORE: 0.2,
        qc_column.ROBUST_Z_SCORE: 0.1,
        qc_column.ROLLING_SCORE: 0.1,
        qc_column.IQR_SCORE: 0.1,
        qc_column.LOF_SCORE: 0.2,
        qc_column.ECDF_SCORE: 0.2,
        qc_column.HAMPEL_SCORE: 0.1
    }
    roll_window = 20
    
    # Create feature normalizer
    normalizer = FeatureNormalizer(features=qc_features)
    
    # Create QC Engine
    qc_engine = QCEngine(
        qc_features=qc_features,
        weights=weights,
        roll_window=roll_window
    )
    
    # Get input path from user
    path = input("Enter full path to PnL_Input.csv: ").strip()
    
    try:
        orchestrator = QCOrchestrator(
            normalizer=normalizer,
            qc_engine=qc_engine,
            input_handler=PnLInput()
        )
        out_path = orchestrator.run(path)
        print(f"\n=== QC Processing Complete ===")
        print(f"Output file: {out_path}")
    except Exception as e:
        logging.getLogger(__name__).exception("QC orchestration failed")
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
