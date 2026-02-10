"""Command-line interface for running QC orchestration."""

import logging
from column_names import pnl_column
from QC_methods.qc_method_definitions import QCMethodDefinitions
from Engine.feature_normalizer import FeatureNormalizer
from Engine.qc_engine import QCEngine
from Engine.score_normalizer import ScoreNormalizer
from qc_orchestrator import QCOrchestrator
from IO.input import PnLInput


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
    
    # Define method configuration (QCMethod -> weight)
    # Only methods listed here will be enabled
    methods_config = {
        QCMethodDefinitions.ISOLATION_FOREST: 0.2,
        QCMethodDefinitions.ROBUST_Z: 0.1,
        QCMethodDefinitions.ROLLING: 0.1,
        QCMethodDefinitions.IQR: 0.1,
        QCMethodDefinitions.LOF: 0.2,
        QCMethodDefinitions.ECDF: 0.2,
        QCMethodDefinitions.HAMPEL: 0.1
    }
    roll_window = 20
    
    # Create feature normalizer
    normalizer = FeatureNormalizer(features=qc_features)
    
    # Create QC Engine
    qc_engine = QCEngine(
        qc_features=qc_features,
        methods_config=methods_config,
        roll_window=roll_window,
        score_normalizer=ScoreNormalizer()
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
