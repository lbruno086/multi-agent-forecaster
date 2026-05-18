# Validation agent operates deterministically via DiagnosticEngine.
# These prompts are used only when the LLM supplements the diagnosis.

VALIDATION_SUPPLEMENT = """Given this model evaluation:
- Fold scores: {fold_scores}
- Train score: {train_score}
- Val score: {val_score}
- Residual stats: {residual_stats}

Does this diagnosis look correct? {diagnosis}
Confidence: {confidence}

Respond with JSON: {{"agree": true/false, "adjusted_confidence": 0.0-1.0, "notes": "..."}}"""
