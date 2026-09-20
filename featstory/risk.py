class RiskDetector:

    def __init__(self, df, target):
        self.df = df
        self.target = target

    def detect(self):

        risks = []

        # Check missing values
        missing = self.df.isnull().sum().sum()

        if missing > 0:
            risks.append(
                f"Dataset contains {missing} missing values."
            )

        # Check duplicate rows
        duplicates = self.df.duplicated().sum()

        if duplicates > 0:
            risks.append(
                f"Dataset contains {duplicates} duplicate rows."
            )

        # Check possible ID columns
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
                risks.append(
                    f"'{column}' may be an identifier column."
                )

        # Check very high correlations
        if self.target in self.df.columns:

            numerical = self.df.select_dtypes(
                include="number"
            )

            correlations = numerical.corr()[self.target]

            for feature, value in correlations.items():

                if feature != self.target and abs(value) >= 0.95:

                    risks.append(
                        f"'{feature}' has a very high "
                        f"correlation with the target "
                        f"({value:.2f}). This is a warning, "
                        f"not proof of data leakage. Check "
                        f"whether this feature is available "
                        f"at prediction time."
                    )                    

        return risks

    def report(self):

        risks = self.detect()

        print("\n===== DATA RISKS =====\n")

        if not risks:
            print("No obvious risks detected.")

        else:
            for risk in risks:
                print(f"- {risk}")