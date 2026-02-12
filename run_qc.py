"""Command-line interface for running QC orchestration."""

import logging
from column_names import pnl_column, main_column
from QC_methods.qc_method_definitions import QCMethodDefinitions
from Engine.feature_normalizer import FeatureNormalizer
from Engine.qc_engine_presets import QCEnginePreset
from qc_orchestrator import QCOrchestrator
from IO.input import PnLInput
from IO.output import Output


def main():
    """Main entry point for QC orchestration."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
    
    # Create QC Engine preset using feature families from column definitions
    engine_preset = QCEnginePreset(
        qc_feature_families=pnl_column.QC_FEATURE_FAMILIES,
        methods_config=methods_config,
        roll_window=roll_window
    )
    
    # Create feature normalizer from all features across families
    normalizer = FeatureNormalizer(features=engine_preset.all_qc_features)
    
    # Get input path from user
    path = input("Enter full path to PnL_Input.csv: ").strip()
    
    try:
        # Create input handler and read/validate data
        input_handler = PnLInput()
        full_data_set = input_handler.read_and_validate(path, split_identifier=main_column.TRADE_TYPE)
        
        # Run orchestration with preprocessed data
        orchestrator = QCOrchestrator(
            normalizer=normalizer,
            engine_preset=engine_preset,
            split_identifier=main_column.TRADE_TYPE
        )
        oos_scores = orchestrator.run(full_data_set)
        
        # Export results to file
        output_handler = Output()
        score_columns = engine_preset.get_score_columns()
        out_path = output_handler.export_full_dataset(
            full_data_set=full_data_set,
            oos_scores=oos_scores,
            input_path=path,
            score_cols=score_columns,
            suffix="_with_scores"
        )
        
        print(f"\n=== QC Processing Complete ===")
        print(f"Output file: {out_path}")
    except Exception as e:
        logging.getLogger(__name__).exception("QC orchestration failed")
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
