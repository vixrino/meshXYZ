"""Root conftest: make the workspace root importable as a package root."""
import sys
import os

# Insert the project root so that `from src.dataset.obj_parser import …` works
# in tests without installing the package.
sys.path.insert(0, os.path.dirname(__file__))
