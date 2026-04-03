from typing import List

from interface.base_datastore import DataItem
from interface.base_indexer import BaseIndexer
from util.bang_rules_parser import parse_bang_rules


class Indexer(BaseIndexer):
    def index(self, document_paths: List[str]) -> List[DataItem]:
        parsed_chunks = parse_bang_rules(document_paths)
        return [DataItem(**chunk) for chunk in parsed_chunks]
