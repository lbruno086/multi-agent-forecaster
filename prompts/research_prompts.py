RESEARCH_SYSTEM = """You are a quantitative finance researcher specializing in time series forecasting.
Your task is to recommend the best predictive methodologies for a financial asset.
Always respond with valid JSON only — no markdown, no explanation outside the JSON."""

RESEARCH_USER = """Find the best methodologies to predict {asset_type} prices for ticker {ticker}.

Context:
- Timeframe: {timeframe}
- Metric to optimize: {metric_name} (lower is better: {is_lower_better})
- Target threshold: {threshold}
- Already tried (do NOT suggest these): {tried_methodologies}

Return a JSON object with this exact schema:
{{
  "asset_type": "{asset_type}",
  "recommended_methodologies": [
    {{
      "name": "skill_name_matching_registry",
      "rank": 1,
      "rationale": "why this approach suits this asset/timeframe",
      "expected_metric_range": {{"low": 0.05, "high": 0.25}},
      "tradeoffs": {{"pros": ["..."], "cons": ["..."]}},
      "implementation_hints": {{"key_param": "value"}},
      "skill_to_use": "xgboost|lightgbm|lstm|arima|prophet"
    }}
  ],
  "feature_engineering_suggestions": ["suggestion1", "suggestion2"],
  "ensemble_strategies": ["strategy1"],
  "sources": [{{"title": "...", "url": "...", "relevance": 0.9}}]
}}

Rank at least 3 methodologies. Focus on approaches proven for {asset_type} {timeframe} forecasting."""
