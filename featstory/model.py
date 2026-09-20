from sklearn.ensemble import RandomForestRegressor

class ModelAnalyzer:

    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.model = None
        self.X = None
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

    def feature_importance(self):

        data = self.df.select_dtypes(
            include="number"
        ).copy()

        id_columns = self.possible_id_columns()

        data = data.drop(
            columns=id_columns,
            errors="ignore"
        )

        X = data.drop(columns=[self.target])
        y = data[self.target]
        self.X=X
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)
        self.model = model
        importance = dict(
            zip(X.columns, model.feature_importances_)
        )

        return dict(
            sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

    def report(self):

        importance = self.feature_importance()

        print("\n===== MODEL FEATURE IMPORTANCE =====\n")

        for feature, value in importance.items():

            print(
                f"{feature}: importance = {value:.3f}"
            )