import pandas as pd

from .profiler import DatasetProfiler
from .analyzer import FeatureAnalyzer
from .visualizer import FeatureVisualizer
from .model import ModelAnalyzer
from .risk import RiskDetector
from .xai import XAIAnalyzer
from .report import ReportGenerator


class Story:

    def __init__(self, data, target=None):

        self.target = target

        if isinstance(data, str):
            self.data = pd.read_csv(data)

        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()

        else:
            raise TypeError(
                "data must be a CSV file path "
                "or pandas DataFrame"
            )

    def summary(self):

        result = {
            "rows": self.data.shape[0],
            "columns": self.data.shape[1],
            "target": self.target,
            "column_names": self.data.columns.tolist()
        }

        return result

    def profile(self):

        profiler = DatasetProfiler(self.data)

        return {
            "basic_info": profiler.basic_info(),
            "data_types": profiler.data_types(),
            "missing_values": profiler.missing_values(),
            "duplicate_rows": profiler.duplicate_rows(),
            "constant_columns": profiler.constant_columns(),
            "high_cardinality_columns": (
                profiler.high_cardinality_columns()
            ),
            "possible_id_columns": (
                profiler.possible_id_columns()
            )
        }

    def analyze(self):

        if not self.target:
            raise ValueError(
                "A target column is required for feature analysis."
            )

        analyzer = FeatureAnalyzer(
            self.data,
            self.target
        )

        return {
            "correlations": analyzer.correlations().to_dict(),
            "feature_stories": analyzer.feature_stories(),
            "categorical_summary": (
                analyzer.categorical_summary()
            )
        }

    def visualize(self):

        if not self.target:
            raise ValueError(
                "A target column is required for visualization."
            )

        analyzer = FeatureAnalyzer(
            self.data,
            self.target
        )

        correlations = analyzer.correlations()

        visualizer = FeatureVisualizer(
            correlations,
            self.target
        )

        return visualizer.correlation_plot()

    def model_importance(self):

        if not self.target:
            raise ValueError(
                "A target column is required for model analysis."
            )

        analyzer = ModelAnalyzer(
            self.data,
            self.target
        )

        return analyzer.feature_importance()

    def detect_risks(self):

        if not self.target:
            raise ValueError(
                "A target column is required for risk detection."
            )

        detector = RiskDetector(
            self.data,
            self.target
        )

        return detector.detect()

    def explain_model(self):

        if not self.target:
            raise ValueError(
                "A target column is required for XAI analysis."
            )

        analyzer = ModelAnalyzer(
            self.data,
            self.target
        )

        analyzer.feature_importance()

        xai = XAIAnalyzer(
            analyzer.model,
            analyzer.X
        )

        return xai.explain()

    def generate_report(
        self,
        filename="featstory_report.html"
    ):

        generator = ReportGenerator(
            self.data,
            self.target
        )

        return generator.generate(filename)