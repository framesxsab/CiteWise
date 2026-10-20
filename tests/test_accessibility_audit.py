from citewise.accessibility_audit import contrast_ratio, audit


def test_contrast_known_black_white_and_equal_colors():
    assert contrast_ratio("#000000", "#ffffff") == 21
    assert contrast_ratio("#ffffff", "#ffffff") == 1


def test_source_audit_finds_missing_control_label(tmp_path):
    path = tmp_path / "fixture.html"
    path.write_text('<html lang="en"><title>Fixture</title><main><h1>Fixture</h1><input id="q"></main></html>')
    result = audit(path)
    assert not result["checks"]["form_controls_have_labels"]
    assert result["missing_labels"] == ["q"]
