import math
import re
from collections import Counter


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        self.documents = {}
        self.term_frequencies = {}
        self.document_frequencies = {}

        self.document_lengths = {}
        self.average_document_length = 0.0

        self.num_documents = 0

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(
            r"\b\w+\b",
            text,
        )

    def add_documents(self, documents):
        self.documents.clear()
        self.term_frequencies.clear()
        self.document_frequencies.clear()
        self.document_lengths.clear()

        for document in documents:
            document_id = document.node_id

            tokens = self._tokenize(document.text)
            term_frequency = Counter(tokens)

            self.documents[document_id] = document
            self.term_frequencies[document_id] = term_frequency
            self.document_lengths[document_id] = len(tokens)

            for term in term_frequency:
                self.document_frequencies[term] = (
                    self.document_frequencies.get(term, 0) + 1
                )

        self.num_documents = len(self.documents)

        if self.num_documents > 0:
            self.average_document_length = (
                sum(self.document_lengths.values()) / self.num_documents
            )

    def _idf(self, term: str) -> float:
        document_frequency = self.document_frequencies.get(term, 0)

        if document_frequency == 0:
            return 0.0

        return math.log(
            1
            + (self.num_documents - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def _score_document(
        self,
        query_terms: list[str],
        document_id: str,
    ) -> float:
        term_frequency = self.term_frequencies[document_id]
        document_length = self.document_lengths[document_id]

        score = 0.0

        for term in query_terms:
            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            idf = self._idf(term)

            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1
                - self.b
                + self.b * (document_length / self.average_document_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        query_terms = self._tokenize(query)

        scores = []

        for document_id in self.documents:
            score = self._score_document(
                query_terms,
                document_id,
            )

            if score > 0:
                scores.append((document_id, score))

        scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scores[:top_k]