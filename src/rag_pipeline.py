from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

from interface import (
    BaseDatastore,
    BaseEvaluator,
    BaseIndexer,
    BaseResponseGenerator,
    BaseRetriever,
    EvaluationResult,
)


@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: Optional[BaseEvaluator] = None
    EVAL_LOG_DIR = Path("logs")

    def reset(self) -> None:
        """Reset the datastore."""
        self.datastore.reset()

    def add_documents(self, documents: List[str]) -> None:
        """Index a list of documents."""
        if not documents:
            print("No source documents found. Skipping indexing.")
            return

        items = self.indexer.index(documents)
        if not items:
            print("No indexable items were extracted from the provided documents.")
            return

        self.datastore.add_items(items)
        print(f"Added {len(items)} items to the datastore.")

    def process_query(self, query: str) -> str:
        search_results = self.retriever.search(query)
        response = self.response_generator.generate_response(query, search_results)
        return response

    def evaluate(
        self, sample_questions: List[Dict[str, str]]
    ) -> List[EvaluationResult]:
        questions = [item["question"] for item in sample_questions]
        expected_answers = [item["answer"] for item in sample_questions]

        with ThreadPoolExecutor(max_workers=10) as executor:
            results: List[EvaluationResult] = list(
                executor.map(
                    self._evaluate_single_question,
                    questions,
                    expected_answers,
                )
            )

        log_path = self._write_evaluation_log(results)

        for i, result in enumerate(results):
            result_label = "OK" if result.is_correct else "FAIL"
            print(f"Q{i+1}: {result_label}")

        number_correct = sum(result.is_correct for result in results)
        print(f"Total Score: {number_correct}/{len(results)}")
        print(f"Evaluation Log: {log_path}")
        return results

    def _evaluate_single_question(
        self, question: str, expected_answer: str
    ) -> EvaluationResult:
        response = self.process_query(question)
        return self.evaluator.evaluate(question, response, expected_answer)

    def _write_evaluation_log(self, results: List[EvaluationResult]) -> str:
        self.EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = self.EVAL_LOG_DIR / f"evaluation-{timestamp}.json"

        payload = {
            "created_at": datetime.now().isoformat(),
            "total_questions": len(results),
            "correct_answers": sum(result.is_correct for result in results),
            "results": [
                {
                    "question": result.question,
                    "response": result.response,
                    "expected_answer": result.expected_answer,
                    "is_correct": result.is_correct,
                    "reasoning": result.reasoning,
                }
                for result in results
            ],
        }

        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(log_path)
