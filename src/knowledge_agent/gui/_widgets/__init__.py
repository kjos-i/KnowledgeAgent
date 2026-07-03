"""Reusable GUI widgets shared across tabs / panels / views.

Kept separate from the top-level GUI modules so a widget's tests
don't drag in the full app-level wiring, and so consumers can import
just what they need.
"""
from knowledge_agent.gui._widgets.resizable_split import ResizableSplit

__all__ = ["ResizableSplit"]
