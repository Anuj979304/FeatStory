import pandas as pd


class FeatureAnalyzer:

    def __init__(self, df, target):
        self.df = df
        self.target = target

    def possible_id_columns(self):

        possible_ids = []

        for column in self.df.columns:

            unique_ratio = self.df[column].nunique() / len(self.df)

            column_name = column.lower()

            name_suggests_id = (
                column_name == "id"
                or column_name.endswith("_id")
                or column_name.endswith("id")
                or "identifier" in column_name
            )

            if unique_ratio == 1 and name_suggests_id:
                possible_ids.append(column)

        return possible_ids

    def categorical_features(self):

        categorical = self.df.select_dtypes(
            exclude="number"
        ).columns.tolist()

        return [
            column
            for column in categorical
            if column not in self.possible_id_columns()
        ]

    def categorical_summary(self):

        results = {}

        features = self.categorical_features()

        for feature in features:

            categories = self.df[feature].dropna().unique()

            target_means = (
                self.df.groupby(feature)[self.target]
                .mean()
                .to_dict()
            )

            results[feature] = {
                "categories": list(categories),
                "target_means": target_means
            }

        return results

    def correlations(self):

        if self.target not in self.df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in dataset."
            )

        numerical_df = self.df.select_dtypes(
            include="number"
        ).copy()

        id_columns = self.possible_id_columns()

        numerical_df = numerical_df.drop(
            columns=id_columns,
            errors="ignore"
        )

        correlations = numerical_df.corr()[self.target]

        correlations = correlations.drop(self.target)

        return correlations.sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )

    def feature_stories(self):

        correlations = self.correlations()

        stories = []

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
            else:
                direction = "negative"

            story = (
                f"{feature}: This feature has a {level} "
                f"{direction} relationship with "
                f"the target '{self.target}'."
            )

            stories.append(story)

        return stories

    def report(self):

        correlations = self.correlations()

        print("\n===== FEATURE-TARGET ANALYSIS =====\n")

        for feature, value in correlations.items():

            print(
                f"{feature}: correlation = {value:.2f}"
            )