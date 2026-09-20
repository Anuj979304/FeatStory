import shap
import matplotlib.pyplot as plt


class XAIAnalyzer:

    def __init__(self, model, X):
        self.model = model
        self.X = X

    def explain(self):

        explainer = shap.TreeExplainer(self.model)

        shap_values = explainer.shap_values(self.X)

        return shap_values

    def plot(self):

        shap_values = self.explain()

        shap.summary_plot(
            shap_values,
            self.X,
            show=True
        )