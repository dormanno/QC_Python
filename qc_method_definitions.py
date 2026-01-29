"""QC Method Definitions using frozen dataclasses for type safety."""

from dataclasses import dataclass
from column_names import qc_column


@dataclass(frozen=True)
class QCMethod:
    """Immutable definition of a QC method.
    
    Attributes:
        name: Internal identifier for the method (used in code)
        score_name: Column name for the score output
    """
    name: str
    score_name: str
    
    def __hash__(self):
        """Make QCMethod hashable for use in dicts and sets."""
        return hash((self.name, self.score_name))


class QCMethods:
    """Collection of all available QC methods as frozen dataclass instances."""
    
    ISOLATION_FOREST = QCMethod(
        name='isolation_forest',
        score_name=qc_column.IF_SCORE
    )
    
    ROBUST_Z = QCMethod(
        name='robust_z',
        score_name=qc_column.ROBUST_Z_SCORE
    )
    
    IQR = QCMethod(
        name='iqr',
        score_name=qc_column.IQR_SCORE
    )
    
    ROLLING = QCMethod(
        name='rolling',
        score_name=qc_column.ROLLING_SCORE
    )
    
    LOF = QCMethod(
        name='lof',
        score_name=qc_column.LOF_SCORE
    )
    
    ECDF = QCMethod(
        name='ecdf',
        score_name=qc_column.ECDF_SCORE
    )
    
    HAMPEL = QCMethod(
        name='hampel',
        score_name=qc_column.HAMPEL_SCORE
    )
    
    @classmethod
    def all_methods(cls) -> list[QCMethod]:
        """Return list of all available QC methods."""
        return [
            cls.ISOLATION_FOREST,
            cls.ROBUST_Z,
            cls.IQR,
            cls.ROLLING,
            cls.LOF,
            cls.ECDF,
            cls.HAMPEL
        ]
    
    @classmethod
    def get_by_name(cls, name: str) -> QCMethod:
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
    def get_name_to_method_mapping(cls) -> dict[str, QCMethod]:
        """Return a dict mapping method names to QCMethod instances."""
        return {method.name: method for method in cls.all_methods()}
