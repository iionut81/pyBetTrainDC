"""Market-specific profiles: hard filters, vetoes, and category scorers.

Each module defines a fixed MarketProfile — the engine core never contains
market-specific thresholds. Adding a new market means adding a new module
here, not touching selection_engine/engine.py.
"""
