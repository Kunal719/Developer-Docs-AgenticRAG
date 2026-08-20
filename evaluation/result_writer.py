import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import PROJECT_VERSION
from evaluation.models import EvaluationResult


class ResultWriter:
    RESULTS_DIRECTORY = Path("evaluation/results")

    @classmethod
    def save(cls, results: list[EvaluationResult]) -> Path:
        """
        Save benchmark results as a timestamped JSON file.
        """

        cls.RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = cls.RESULTS_DIRECTORY / f"{PROJECT_VERSION}_{timestamp}.json"

        serialized_results = []

        for result in results:
            result_dict = asdict(result)

            if result.judge_evaluation is not None:
                result_dict["judge_evaluation"] = (
                    result.judge_evaluation.model_dump()
                )
            else:
                result_dict["judge_evaluation"] = None

            # Convert datetime to ISO string
            result_dict["experiment_metadata"]["timestamp"] = (
                result.experiment_metadata.timestamp.isoformat()
            )

            serialized_results.append(result_dict)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(serialized_results, file, indent=4, ensure_ascii=False)

        return output_path