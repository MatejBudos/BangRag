import math
import re

from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever


class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 10) -> list[str]:
        search_results = self.datastore.search(query, top_k=top_k * 3)
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(
        self, query: str, search_results: list[str], top_k: int = 10
    ) -> list[str]:
        scored_results = []
        query_terms = self._tokenize(query)
        idf = self._inverse_document_frequency(query_terms, search_results)

        for index, document in enumerate(search_results):
            lexical_score = self._score_document(query_terms, idf, document)
            vector_rank_score = 1.0 / (index + 1)
            general_rules_bonus = self._general_rules_bonus(query_terms, document)
            final_score = (vector_rank_score * 3.0) + lexical_score + general_rules_bonus
            scored_results.append((final_score, index))

        scored_results.sort(key=lambda item: item[0], reverse=True)
        result_indices = [index for _, index in scored_results[:top_k]]
        return [search_results[i] for i in result_indices]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold())

    def _inverse_document_frequency(
        self, query_terms: list[str], documents: list[str]
    ) -> dict[str, float]:
        if not documents:
            return {}

        idf = {}
        tokenized_documents = [self._tokenize(document) for document in documents]
        for term in set(query_terms):
            doc_freq = sum(1 for document in tokenized_documents if term in document)
            idf[term] = math.log((1 + len(documents)) / (1 + doc_freq)) + 1.0
        return idf

    def _score_document(
        self, query_terms: list[str], idf: dict[str, float], document: str
    ) -> float:
        document_terms = self._tokenize(document)
        if not document_terms:
            return 0.0

        term_frequencies: dict[str, int] = {}
        for term in document_terms:
            term_frequencies[term] = term_frequencies.get(term, 0) + 1

        score = 0.0
        document_length = len(document_terms)
        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            score += (term_frequency / document_length) * idf.get(term, 1.0)

        return score

    def _general_rules_bonus(self, query_terms: list[str], document: str) -> float:
        lowered_document = document.casefold()
        bonus = 0.0

        is_general_rules_chunk = "vseobecne pravidla:" in lowered_document
        is_card_effect_chunk = "karta a efekt" in lowered_document

        if is_general_rules_chunk:
            bonus += 0.8

        if is_card_effect_chunk:
            bonus += 1.2

        card_effect_terms = {
            "karta",
            "karty",
            "efekt",
            "efekty",
            "pivo",
            "bang",
            "vedle",
            "limit",
            "zakaz",
            "zákaz",
            "moze",
            "môže",
            "nemoze",
            "nemôže",
        }
        if (is_general_rules_chunk or is_card_effect_chunk) and any(
            term in card_effect_terms for term in query_terms
        ):
            bonus += 1.0

        return bonus
