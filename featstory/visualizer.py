import matplotlib.pyplot as plt


class FeatureVisualizer:

    def __init__(self, correlations, target):
        self.correlations = correlations
        self.target = target

    def correlation_plot(self):
        data = self.correlations.sort_values()

        plt.figure(figsize=(8, 5))

        data.plot(kind="barh")

        plt.title(
            f"Feature Relationships with '{self.target}'"
        )

        plt.xlabel("Correlation")
        plt.ylabel("Features")

        plt.tight_layout()

        plt.show()