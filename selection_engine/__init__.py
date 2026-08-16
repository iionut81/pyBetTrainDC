"""Generic filter + ranking engine for betting market candidate selection.

This package is intentionally separate from the tuned production models
(dc_double_chance.py, wta_set1_filters.py, run_wta_daily.py, config.yaml).
It does not predict outcomes or price markets — given N candidate matches for
a chosen market, it eliminates the weak ones (with a reason), scores what's
left, and returns a ranked TOP N, or NO_BET if nothing clears the bar.
"""
