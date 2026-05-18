from __future__ import annotations

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from evaluation.models import DiagnosticReport
from tools.logger import get_logger

log = get_logger(__name__)


class DiagnosticEngine:
    """Diagnoses why a model underperforms by analyzing training signals."""

    OVERFITTING_GAP = 0.15
    HIGH_VARIANCE_RATIO = 0.30
    SLOW_IMPROVEMENT_THRESHOLD = 0.01
    LJUNGBOX_PVALUE = 0.05

    def diagnose(
        self,
        metric_name: str,
        metric_value: float,
        train_score: float,
        val_score: float,
        fold_scores: list[float],
        residuals: np.ndarray | None = None,
    ) -> DiagnosticReport:
        evidence: dict = {}
        signals: list[tuple[str, float]] = []

        # Signal 1: overfitting — train much better than val
        gap = abs(train_score - val_score)
        evidence["train_val_gap"] = round(gap, 4)
        if gap > self.OVERFITTING_GAP:
            signals.append(("PARAMETER_ISSUE", 0.75 + min(gap * 0.5, 0.20)))

        # Signal 2: flat / high loss on both — model can't capture signal
        # Threshold assumes MAPE/proportional_error scale (0–1). Not valid for MSE/R².
        if train_score > 0.40 and val_score > 0.40:
            flatness = 1.0 - abs(train_score - val_score) / max(train_score, 1e-9)
            evidence["both_high_loss"] = True
            signals.append(("METHODOLOGY_ISSUE", 0.60 + flatness * 0.25))

        # Signal 3: high variance across folds — architecture unstable
        if len(fold_scores) >= 2:
            fold_arr = np.array(fold_scores)
            fold_std = float(np.std(fold_arr))
            fold_mean = float(np.mean(fold_arr))
            variance_ratio = fold_std / max(fold_mean, 1e-9)
            evidence["fold_variance_ratio"] = round(variance_ratio, 4)
            if variance_ratio > self.HIGH_VARIANCE_RATIO:
                signals.append(("ARCHITECTURE_ISSUE", 0.55 + min(variance_ratio * 0.3, 0.35)))

        # Signal 4: residual autocorrelation — model misses temporal structure
        if residuals is not None and len(residuals) >= 10:
            try:
                lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
                pvalue = float(lb["lb_pvalue"].iloc[0])
                evidence["ljungbox_pvalue"] = round(pvalue, 4)
                if pvalue < self.LJUNGBOX_PVALUE:
                    signals.append(("METHODOLOGY_ISSUE", 0.70 + (1 - pvalue) * 0.25))
            except Exception as exc:
                log.warning("ljungbox_failed", error=str(exc))

        # Signal 5: slow improvement across folds
        if len(fold_scores) >= 3:
            early = np.mean(fold_scores[: len(fold_scores) // 2])
            late = np.mean(fold_scores[len(fold_scores) // 2 :])
            improvement = abs(early - late) / max(early, 1e-9)
            evidence["fold_improvement_rate"] = round(improvement, 4)
            if improvement < self.SLOW_IMPROVEMENT_THRESHOLD:
                signals.append(("PARAMETER_ISSUE", 0.50 + improvement * 5))

        root_cause, confidence = self._resolve(signals)

        details = self._build_details(root_cause, evidence, train_score, val_score)

        return DiagnosticReport(
            metric_value=metric_value,
            metric_name=metric_name,
            root_cause=root_cause,
            evidence=evidence,
            confidence=round(min(confidence, 1.0), 3),
            details=details,
            fold_scores=fold_scores,
            train_score=train_score,
            val_score=val_score,
        )

    def _resolve(
        self, signals: list[tuple[str, float]]
    ) -> tuple[str, float]:
        if not signals:
            return "PARAMETER_ISSUE", 0.40

        by_cause: dict[str, list[float]] = {}
        for cause, conf in signals:
            by_cause.setdefault(cause, []).append(conf)

        scores = {
            cause: max(confs) for cause, confs in by_cause.items()
        }
        best = max(scores, key=lambda k: scores[k])
        return best, scores[best]

    def _build_details(
        self,
        root_cause: str,
        evidence: dict,
        train_score: float,
        val_score: float,
    ) -> str:
        if root_cause == "PARAMETER_ISSUE":
            return (
                f"Model shows parameter-level issues. "
                f"train={train_score:.3f}, val={val_score:.3f}. "
                f"Recommendation: run Optuna hyperparameter tuning."
            )
        if root_cause == "ARCHITECTURE_ISSUE":
            return (
                f"High variance across folds suggests the architecture "
                f"doesn't generalize well to different market regimes. "
                f"Recommendation: try a different model variant or feature set."
            )
        return (
            f"The model family cannot capture the underlying signal. "
            f"train={train_score:.3f}, val={val_score:.3f}. "
            f"Recommendation: research a different methodology."
        )
