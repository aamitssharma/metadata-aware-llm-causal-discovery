"""
LLM module facade for the OpenRouter client.
"""
from typing import Literal

from client import OpenRouterLLM

QueryMode = Literal["edge", "no_edge"]

__all__ = ["OpenRouterLLM", "QueryMode"]
