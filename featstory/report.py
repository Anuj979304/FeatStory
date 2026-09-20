from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class ReportGenerator:

    def __init__(self, df, target):
        self.df = df
        self.target = target

    def generate(
        self,
        filename="featstory_report.html"
    ):

        rows = self.df.shape[0]
        columns = self.df.shape[1]

        # =========================================================
        # FIND POSSIBLE ID COLUMNS
        # =========================================================

        id_columns = []

        for column in self.df.columns:

            unique_ratio = (
                self.df[column].nunique() / len(self.df)
            )

            column_name = column.lower()

            name_suggests_id = (
                column_name == "id"
                or column_name.endswith("_id")
                or column_name.endswith("id")
                or "identifier" in column_name
            )

            if unique_ratio == 1 and name_suggests_id:
                id_columns.append(column)

        # =========================================================
        # MODEL FEATURE IMPORTANCE
        # =========================================================

        model_data = self.df.select_dtypes(
            include="number"
        ).copy()

        model_data = model_data.drop(
            columns=id_columns,
            errors="ignore"
        )

        X = model_data.drop(
            columns=[self.target]
        )

        y = model_data[self.target]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        importance = dict(
            zip(
                X.columns,
                model.feature_importances_
            )
        )

        importance = dict(
            sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        # =========================================================
        # CREATE MODEL FEATURE IMPORTANCE CHART
        # =========================================================

        features = list(importance.keys())
        values = list(importance.values())

        plt.figure(figsize=(8, 5))

        plt.barh(
            features[::-1],
            values[::-1]
        )

        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Model Feature Importance")

        plt.tight_layout()

        plt.savefig(
            "featstory_feature_importance.png"
        )

        plt.close()

        # =========================================================
        # CORRELATION ANALYSIS
        # =========================================================

        numerical = self.df.select_dtypes(
            include="number"
        ).copy()

        numerical = numerical.drop(
            columns=id_columns,
            errors="ignore"
        )

        correlations = numerical.corr()[self.target]

        correlations = correlations.drop(
            self.target
        )

        # =========================================================
        # CATEGORICAL COLUMNS
        # =========================================================

        categorical_columns = self.df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        categorical_columns = [
            column
            for column in categorical_columns
            if column not in id_columns
        ]

        # =========================================================
        # DATA RISKS
        # =========================================================

        risks = []

        # Missing values
        missing = self.df.isnull().sum().sum()

        if missing > 0:

            risks.append(
                f"Dataset contains {missing} missing values."
            )

        # Duplicate rows
        duplicates = self.df.duplicated().sum()

        if duplicates > 0:

            risks.append(
                f"Dataset contains {duplicates} duplicate rows."
            )

        # High-cardinality categorical columns
        for column in categorical_columns:

            unique_ratio = (
                self.df[column].nunique() / len(self.df)
            )

            if unique_ratio >= 0.8:

                risks.append(
                    f"'{column}' has high cardinality. "
                    f"Many unique categories may make this "
                    f"feature difficult to use effectively."
                )

        # Possible ID columns
        for column in id_columns:

            risks.append(
                f"'{column}' may be an identifier column."
            )

        # Very high correlations
        for feature, value in correlations.items():

            if abs(value) >= 0.95:

                risks.append(
                    f"'{feature}' has a very high "
                    f"correlation with the target "
                    f"({value:.2f}). This is a warning, "
                    f"not proof of data leakage. Check "
                    f"whether this feature is available "
                    f"at prediction time."
                )

        # =========================================================
        # CREATE RISK HTML
        # =========================================================

        risk_rows = ""

        if risks:

            for risk in risks:

                risk_rows += f"""
                <li>{risk}</li>
                """

        else:

            risk_rows = """
            <li>No obvious risks detected.</li>
            """

        # =========================================================
        # CREATE CATEGORICAL ANALYSIS TABLE
        # =========================================================

        categorical_rows = ""

        for feature in categorical_columns:

            target_means = (
                self.df.groupby(feature)[self.target]
                .mean()
            )

            for category, mean_value in target_means.items():

                categorical_rows += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{category}</td>
                    <td>{mean_value:.2f}</td>
                </tr>
                """

        # =========================================================
        # CREATE CATEGORICAL CHARTS
        # =========================================================

        categorical_chart_html = ""

        for feature in categorical_columns:

            target_means = (
                self.df.groupby(feature)[self.target]
                .mean()
            )

            plt.figure(figsize=(7, 4))

            target_means.plot(
                kind="bar"
            )

            plt.title(
                f"{feature} vs {self.target}"
            )

            plt.xlabel(feature)

            plt.ylabel(
                f"Average {self.target}"
            )

            plt.tight_layout()

            chart_filename = (
                f"featstory_{feature}_target.png"
            )

            plt.savefig(
                chart_filename
            )

            plt.close()

            categorical_chart_html += f"""
            <div class="chart">
                <h3>{feature}</h3>

                <img
                    src="{chart_filename}"
                    alt="{feature} vs {self.target}"
                    style="width: 100%; max-width: 700px;"
                >
            </div>
            """

        if not categorical_chart_html:

            categorical_chart_html = """
            <p>No categorical feature charts available.</p>
            """

        # =========================================================
        # CREATE FEATURE STORIES
        # =========================================================

        feature_stories = ""

        for feature, value in correlations.items():

            strength = abs(value)

            if strength >= 0.8:
                level = "very strong"

            elif strength >= 0.6:
                level = "strong"

            elif strength >= 0.4:
                level = "moderate"

            elif strength >= 0.2:
                level = "weak"

            else:
                level = "very weak"

            if value > 0:

                direction = "positive"

                explanation = (
                    "Higher values of this feature are "
                    "associated with higher target values."
                )

            else:

                direction = "negative"

                explanation = (
                    "Higher values of this feature are "
                    "associated with lower target values."
                )

            feature_stories += f"""
            <li>
                <strong>{feature}</strong> has a
                {level} {direction} relationship
                with the target <strong>{self.target}</strong>
                (correlation: <strong>{value:.2f}</strong>).

                {explanation}
            </li>
            """

        # =========================================================
        # CREATE MODEL IMPORTANCE TABLE
        # =========================================================

        importance_rows = ""

        for feature, value in importance.items():

            importance_rows += f"""
            <tr>
                <td>{feature}</td>
                <td>{value:.3f}</td>
            </tr>
            """

        # =========================================================
        # CREATE CORRELATION TABLE
        # =========================================================

        correlation_rows = ""

        for feature, value in correlations.items():

            correlation_rows += f"""
            <tr>
                <td>{feature}</td>
                <td>{value:.2f}</td>
            </tr>
            """

        # =========================================================
        # HTML REPORT
        # =========================================================

        html = f"""
<!DOCTYPE html>

<html>

<head>

    <title>FeatStory Report</title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f7fa;
        }}

        .container {{
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
        }}

        h1 {{
            color: #1f4e79;
        }}

        h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }}

        h3 {{
            color: #1f4e79;
        }}

        .card {{
            background: #eef4f8;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
        }}

        .chart {{
            background: #eef4f8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 6px;
            text-align: center;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        th,
        td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}

        th {{
            background-color: #1f4e79;
            color: white;
        }}

        img {{
            border-radius: 6px;
        }}

    </style>

</head>

<body>

<div class="container">

    <h1>FeatStory Report</h1>

    <p>
        Generated on:
        {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </p>

    <!-- DATASET OVERVIEW -->

    <h2>Dataset Overview</h2>

    <div class="card">
        <strong>Rows:</strong> {rows}
    </div>

    <div class="card">
        <strong>Columns:</strong> {columns}
    </div>

    <div class="card">
        <strong>Target:</strong> {self.target}
    </div>

    <!-- DATA RISKS -->

    <h2>Data Risks</h2>

    <div class="card">

        <ul>
            {risk_rows}
        </ul>

    </div>

    <!-- CATEGORICAL CHARTS -->

    <h2>Categorical Feature Charts</h2>

    {categorical_chart_html}

    <!-- CATEGORICAL ANALYSIS -->

    <h2>Categorical Feature Analysis</h2>

    <table>

        <tr>
            <th>Feature</th>
            <th>Category</th>
            <th>Average Target</th>
        </tr>

        {categorical_rows}

    </table>

    <!-- FEATURE STORIES -->

    <h2>Feature Stories</h2>

    <div class="card">

        <ul>
            {feature_stories}
        </ul>

    </div>

    <!-- MODEL IMPORTANCE CHART -->

    <h2>Model Feature Importance Chart</h2>

    <div class="chart">

        <img
            src="featstory_feature_importance.png"
            alt="Model Feature Importance"
            style="width: 100%; max-width: 800px;"
        >

    </div>

    <!-- MODEL IMPORTANCE TABLE -->

    <h2>Model Feature Importance</h2>

    <table>

        <tr>
            <th>Feature</th>
            <th>Importance</th>
        </tr>

        {importance_rows}

    </table>

    <!-- CORRELATIONS -->

    <h2>Feature Correlations</h2>

    <table>

        <tr>
            <th>Feature</th>
            <th>Correlation</th>
        </tr>

        {correlation_rows}

    </table>

</div>

</body>

</html>
"""

        # =========================================================
        # SAVE REPORT
        # =========================================================

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(html)

        print(
            f"\nReport generated successfully: {filename}"
        )