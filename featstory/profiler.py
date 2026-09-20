import pandas as pd


class DatasetProfiler:

    def __init__(self, df):
        self.df = df

    def basic_info(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns)
        }

    def data_types(self):
        numerical = []
        categorical = []

        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                numerical.append(column)
            else:
                categorical.append(column)

        return {
            "numerical": numerical,
            "categorical": categorical
        }

    def missing_values(self):

        missing = self.df.isnull().sum()

        return missing[missing > 0]    

    def duplicate_rows(self):
        return self.df.duplicated().sum()

    def constant_columns(self):

        constant = []

        for column in self.df.columns:
            if self.df[column].nunique(dropna=False) <= 1:
                constant.append(column)

        return constant
    
    def high_cardinality_columns(self):

        high_cardinality = []

        for column in self.df.select_dtypes(
            exclude="number"
        ).columns:

            unique_ratio = (
                self.df[column].nunique() / len(self.df)
            )

            if unique_ratio >= 0.8:
                high_cardinality.append(column)

        return high_cardinality    

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

    def report(self):

        info = self.basic_info()
        types = self.data_types()

        print("\n===== DATASET PROFILE =====\n")

        print(f"Rows: {info['rows']}")
        print(f"Columns: {info['columns']}")

        print(f"\nNumerical columns: {len(types['numerical'])}")
        print(f"Categorical columns: {len(types['categorical'])}")

        missing_values = self.missing_values()

        print(f"\nMissing values: {missing_values.sum()}")

        print("\nMissing values by column:")

        if missing_values.empty:
            print("None")
        else:
            for column, count in missing_values.items():
                print(f"- {column}: {count}")  

                print(f"Duplicate rows: {self.duplicate_rows()}")
                constants = self.constant_columns()
        constants = self.constant_columns()
        print("\nConstant columns:")
        if constants:
            for column in constants:
                print(f"- {column}")
        else:
            print("None")

            high_cardinality = self.high_cardinality_columns()

        print("\nHigh-cardinality categorical columns:")

        if high_cardinality:
            for column in high_cardinality:
                print(f"- {column}")
        else:
            print("None")        
        ids = self.possible_id_columns()

        print("\nPossible ID columns:")

        if ids:
            for column in ids:
                print(f"- {column}")
        else:
            print("None")