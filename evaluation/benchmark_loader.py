# from evaluation.models import BenchmarkQuestion
from models.query_input import BenchmarkQuestion
import json
from pathlib import Path

def load_questions(path: Path) -> list[BenchmarkQuestion]:
    """
    Load benchmark questions from a JSON file.

    Args:
        path: The path to the JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [BenchmarkQuestion(**question) for question in json.load(f)]