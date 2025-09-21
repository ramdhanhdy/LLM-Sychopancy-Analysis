"""Flask dashboard application factory."""
from .app import app, create_app  # noqa: F401

__all__ = ['app', 'create_app']
