# FeatStory

### Don't just know which features matter. Understand why.

FeatStory is a Python data science library that helps you understand how features in a dataset relate to a target variable.

Instead of only showing numbers, FeatStory creates simple **feature stories**, identifies possible data risks, analyzes categorical features, calculates model-based feature importance, and generates visual reports.

## Features

* Dataset profiling
* Missing-value detection
* Duplicate-row detection
* Constant-column detection
* Possible ID-column detection
* High-cardinality detection
* Feature-target correlation analysis
* Simple feature stories
* Categorical feature analysis
* Data-risk warnings
* Random Forest feature importance
* SHAP-based model explanation
* Correlation visualizations
* HTML report generation
* Automated tests with pytest

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn shap pytest
```

## Basic Usage

```python
from featstory import Story

story = Story(
    "student_data.csv",
    target="final_score"
)

print(story.summary())

print(story.profile())

print(story.analyze())

print(story.detect_risks())

print(story.model_importance())

story.visualize()

story.explain_model()

story.generate_report()
```

## Example Dataset

FeatStory can analyze datasets containing numerical and categorical features.

Example:

```text
student_id
gender
study_hours
attendance
assignments
previous_score
final_score
```

For example, FeatStory can identify relationships such as:

```text
study_hours → final_score
attendance → final_score
previous_score → final_score
```

It can also compare the target across categorical groups such as gender.

## Project Structure

```text
FeatStory/
│
├── featstory/
│   ├── __init__.py
│   ├── story.py
│   ├── profiler.py
│   ├── analyzer.py
│   ├── visualizer.py
│   ├── model.py
│   ├── risk.py
│   ├── xai.py
│   └── report.py
│
├── examples/
│   ├── __init__.py
│   ├── test_story.py
│   └── student_data.csv
│
├── tests/
│   └── test_featstory.py
│
└── README.md
```

## Running Tests

From the project folder:

```bash
python -m pytest
```

The project currently includes automated tests covering the main library functionality.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* SHAP
* Pytest

## Project Goal

FeatStory is designed as an educational data science tool that helps users move beyond simply seeing feature importance and understand the relationships, patterns, and possible risks within their datasets.

## Status

FeatStory is currently under active development.
