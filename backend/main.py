"""Compatibility entry point for the same application as cash_ml.api:app."""

from backend.app_factory import create_app as create_app
from cash_ml.api import app as app
