import numpy as np

from evaluation.diagnostic_engine import DiagnosticEngine


def test_overfitting_detected_as_parameter_issue():
    engine = DiagnosticEngine()
    report = engine.diagnose(
        metric_name="mape",
        metric_value=0.35,
        train_score=0.10,   # much better than val
        val_score=0.35,
        fold_scores=[0.33, 0.35, 0.36, 0.34, 0.35],
    )
    assert report.root_cause == "PARAMETER_ISSUE"
    assert report.confidence > 0.5


def test_flat_high_loss_is_methodology_issue():
    engine = DiagnosticEngine()
    report = engine.diagnose(
        metric_name="mape",
        metric_value=0.60,
        train_score=0.58,
        val_score=0.62,
        fold_scores=[0.59, 0.61, 0.60, 0.62, 0.61],
    )
    assert report.root_cause == "METHODOLOGY_ISSUE"


def test_high_fold_variance_is_architecture_issue():
    engine = DiagnosticEngine()
    report = engine.diagnose(
        metric_name="mape",
        metric_value=0.35,
        train_score=0.30,
        val_score=0.35,
        fold_scores=[0.10, 0.60, 0.10, 0.70, 0.10],  # extreme variance
    )
    assert report.root_cause == "ARCHITECTURE_ISSUE"


def test_autocorrelated_residuals_is_methodology_issue():
    engine = DiagnosticEngine()
    # Strongly autocorrelated residuals (trending error)
    residuals = np.cumsum(np.ones(50) * 0.1)
    report = engine.diagnose(
        metric_name="mape",
        metric_value=0.35,
        train_score=0.32,
        val_score=0.35,
        fold_scores=[0.34, 0.35, 0.35, 0.36, 0.35],
        residuals=residuals,
    )
    assert report.root_cause == "METHODOLOGY_ISSUE"


def test_report_has_all_fields():
    engine = DiagnosticEngine()
    report = engine.diagnose(
        metric_name="mape",
        metric_value=0.35,
        train_score=0.30,
        val_score=0.35,
        fold_scores=[0.33, 0.35, 0.36],
    )
    assert 0.0 <= report.confidence <= 1.0
    assert report.metric_name == "mape"
    assert isinstance(report.details, str)
    assert isinstance(report.evidence, dict)
