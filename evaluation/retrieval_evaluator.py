from evaluation.models import  PipelineStage, RetrievedDocument
from models.query_input import BenchmarkQuestion

class RetrievalEvaluator:

    def evaluate(self, question: BenchmarkQuestion, stages: list[PipelineStage]) -> dict[str, float]:
        """
        Returns the overall metrics

        Args:
            question: The benchmark question
            stages: The pipeline stages
        """
        retrieved_metrics = {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0
        }

        rrf_documents : list[RetrievedDocument] = []

        for stage in stages:
            # Calculate Recall@K when stage is RRF Fusion
            if stage.name in ("RRF Fusion", "Multi Query Fusion"):
                rrf_documents.extend(stage.documents)

            elif stage.name == "Reranker":
                precision_at_k = self._calculate_precision_at_k(question.expected_files, stage.documents)
                retrieved_metrics["precision_at_k"] = precision_at_k

                hit_rate = self._calculate_hit_rate(question.expected_files, stage.documents)
                retrieved_metrics["hit_rate"] = hit_rate

                mrr = self._calculate_mrr(question.expected_files, stage.documents)
                retrieved_metrics["mrr"] = mrr

        if rrf_documents:
            recall_at_k = self._calculate_recall_at_k(question.expected_files, rrf_documents)
            retrieved_metrics["recall_at_k"] = recall_at_k

        return retrieved_metrics

    def _get_relevant_retrieved_files(self, expected_files: list[str], retrieved_documents: list[RetrievedDocument]) -> set[str]:
        expected = set(expected_files)
        retrieved = {doc.source for doc in retrieved_documents}
        return expected.intersection(retrieved)

      
    def _calculate_recall_at_k(self, expected_files: list[str], retrieved_documents: list[RetrievedDocument]) -> float:
        """
        Calculate Recall@K

        Args:
            expected_files: The expected files
            retrieved_documents: The retrieved documents
        """
        relevant_retrieved_docs = self._get_relevant_retrieved_files(expected_files, retrieved_documents)

        return len(relevant_retrieved_docs) / len(expected_files)

    def _calculate_precision_at_k(self, expected_files: list[str], retrieved_documents: list[RetrievedDocument]) -> float:
        """
        Calculate Precision@K

        Args:
            expected_files: The expected files
            retrieved_documents: The retrieved documents
        """
        relevant_retrieved_docs = self._get_relevant_retrieved_files(expected_files, retrieved_documents)

        return len(relevant_retrieved_docs) / len(set(doc.source for doc in retrieved_documents))

    def _calculate_hit_rate(self, expected_files: list[str], retrieved_documents: list[RetrievedDocument]) -> float:
        """
        Calculate Hit Rate

        Args:
            expected_files: The expected files
            retrieved_documents: The retrieved documents
        """
        relevant_retrieved_docs = self._get_relevant_retrieved_files(expected_files, retrieved_documents)

        return 1.0 if len(relevant_retrieved_docs) > 0 else 0.0

    def _calculate_mrr(self, expected_files: list[str], retrieved_documents: list[RetrievedDocument]) -> float:
        """
        Calculate MRR

        Args:
            expected_files: The expected files
            retrieved_documents: The retrieved documents
        """

        for doc in retrieved_documents:
            if doc.source in expected_files:
                return 1.0 / doc.rank

        return 0.0