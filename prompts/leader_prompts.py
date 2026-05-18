LEADER_SYSTEM = """You are the orchestrator of a multi-agent financial forecasting system.
You receive diagnostic reports and decide the next action.
Respond with valid JSON only."""

LEADER_DECISION_USER = """Current workflow state:
- Ticker: {ticker} | Timeframe: {timeframe}
- Metric: {metric_name} | Target: {threshold}
- Outer iteration: {outer_iter}/{max_outer}
- Inner attempt: {inner_attempt}/{max_inner}
- Current metric value: {current_metric}
- Best achieved: {best_metric}
- Tried methodologies: {tried_methodologies}

Diagnostic report:
- Root cause: {root_cause}
- Confidence: {confidence}
- Details: {details}
- Evidence: {evidence}

Decide the next action. Return JSON:
{{
  "action": "TUNE|NEW_VARIANT|RESEARCH|SUCCESS|FAILED",
  "reason": "brief explanation",
  "suggested_params": {{}}
}}"""
