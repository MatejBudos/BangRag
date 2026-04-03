import os
import sys
from typing import List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from create_parser import create_parser
from impl import Datastore, Evaluator, Indexer, ResponseGenerator, Retriever
from rag_pipeline import RAGPipeline
from util.faq_loader import load_evaluation_questions


DEFAULT_SOURCE_PATH = "BangRules"
DEFAULT_EVAL_PATH = "FAQ"
SUPPORTED_SOURCE_EXTENSIONS = {".tex"}


def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)


def main():
    parser = create_parser()
    args = parser.parse_args()
    pipeline = create_pipeline()

    source_path = args.path if args.path else DEFAULT_SOURCE_PATH
    eval_path = args.eval_file if args.eval_file else DEFAULT_EVAL_PATH
    document_paths = get_files_in_directory(source_path)

    if args.command in ["reset", "run"]:
        print("Resetting the database...")
        pipeline.reset()

    if args.command in ["add", "run"]:
        print(f"Adding documents: {', '.join(document_paths)}")
        pipeline.add_documents(document_paths)

    if args.command in ["evaluate", "run"]:
        print(f"Evaluating using questions from: {eval_path}")
        sample_questions = load_evaluation_questions(eval_path)
        pipeline.evaluate(sample_questions)

    if args.command == "query":
        print(f"Response: {pipeline.process_query(args.prompt)}")


def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]

    document_paths = []
    for root, _, files in os.walk(source_path):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in SUPPORTED_SOURCE_EXTENSIONS:
                document_paths.append(os.path.join(root, file_name))
    return sorted(document_paths)


if __name__ == "__main__":
    main()
