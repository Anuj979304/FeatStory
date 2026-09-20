from featstory import Story

story = Story(
    "examples/student_data.csv",
    target="final_score"
)

story.summary()

story.profile()

story.analyze()

story.visualize()

story.model_importance()

story.detect_risks()

story.explain_model()

story.generate_report()