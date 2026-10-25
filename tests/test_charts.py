from citewise.charts import _paired_for, scope_label


def test_chart_scope_marks_partial_results_as_pilot():
    rows = [{"questions_evaluated": 40}]
    assert scope_label({"split": "validation", "coverage": {"scorable": 888}}, rows) == \
        "QASPER validation PILOT | 40 scorable questions"
    assert scope_label({"split": "validation", "limit": 40}, rows) == \
        "QASPER validation PILOT | 40 scorable questions"
    assert scope_label({"split": "validation", "coverage": {"scorable": 888}},
                       [{"questions_evaluated": 888}]) == "QASPER validation | 888 scorable questions"
    assert scope_label({"split": "test"}, [{"questions_evaluated": 1309}]) == "QASPER test | 1309 scorable questions"


def test_paired_chart_orients_e5_minus_comparator():
    comparisons = [{"model_a": "bge-base", "model_b": "e5-base-v2", "metric": "recall_at_5",
                    "mean_difference_a_minus_b": -.0257, "ci95_low": -.0526, "ci95_high": -.0004}]
    assert _paired_for("e5-base-v2", "bge-base", "recall_at_5", comparisons) == (.0257, .0004, .0526)
