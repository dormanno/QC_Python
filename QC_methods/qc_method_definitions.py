"""QC Method Definitions using frozen dataclasses for type safety."""

from dataclasses import dataclass
from column_names import qc_column


@dataclass(frozen=True)
class QCMethodDefinition:
    """Immutable definition of a QC method.
    
    Attributes:
        name: Internal identifier for the method (used in code)
        score_name: Column name for the score output
    """
    name: str
    score_name: str
    
    def __hash__(self):
        """Make QCMethodDefinition hashable for use in dicts and sets."""
        return hash((self.name, self.score_name))


class QCMethodDefinitions:
    """Collection of all available QC methods as frozen dataclass instances."""
    
    ISOLATION_FOREST = QCMethodDefinition(
        name='isolation_forest',
        score_name=qc_column.IF_SCORE
    )
    
    ROBUST_Z = QCMethodDefinition(
        name='robust_z',
        score_name=qc_column.ROBUST_Z_SCORE
    )
    
    IQR = QCMethodDefinition(
        name='iqr',
        score_name=qc_column.IQR_SCORE
    )
    
    ROLLING = QCMethodDefinition(
        name='rolling',
        score_name=qc_column.ROLLING_SCORE
    )
    
    LOF = QCMethodDefinition(
        name='lof',
        score_name=qc_column.LOF_SCORE
    )
    
    ECDF = QCMethodDefinition(
        name='ecdf',
        score_name=qc_column.ECDF_SCORE
    )
    
    HAMPEL = QCMethodDefinition(
        name='hampel',
        score_name=qc_column.HAMPEL_SCORE
    )
    
    STALE_VALUE = QCMethodDefinition(
        name='stale_value',
        score_name=qc_column.STALE_SCORE
    )
    
    @classmethod
    def all_methods(cls) -> list[QCMethodDefinition]:
        """Return list of all available QC methods."""
        return [
            cls.ISOLATION_FOREST,
            cls.ROBUST_Z,
            cls.IQR,
            cls.ROLLING,
            cls.LOF,
            cls.ECDF,
            cls.HAMPEL,
            cls.STALE_VALUE
        ]
    
    @classmethod
    def get_by_name(cls, name: str) -> QCMethodDefinition:
        """Get a QCMethod by its name string.
        
        Args:
            name: The method name string
            
        Returns:
            QCMethod instance
            
        Raises:
            ValueError: If method name is not found
        """
        for method in cls.all_methods():
            if method.name == name:
                return method
        raise ValueError(
            f"Unknown method name: '{name}'. "
            f"Valid options: {[m.name for m in cls.all_methods()]}"
        )
    
    @classmethod
    def get_name_to_method_mapping(cls) -> dict[str, QCMethodDefinition]:
        """Return a dict mapping method names to QCMethod instances."""
        return {method.name: method for method in cls.all_methods()}
