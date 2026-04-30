from beir.retrieval.models.hyde import HyDE, OpenAIHypothesisGenerator, HyDEPromptBuilder

class BM25WithHyDE:
    def __init__(self, corpus, hyde: HyDE, k1=1.2, b=0.75):
        """
        Initialize BM25 with HyDE query expansion.
        :param corpus: List of documents, where each document is a list of terms.
        :param hyde: HyDE instance for query expansion.
        :param k1: Term frequency saturation parameter.
        :param b: Length normalization parameter.
        """
        self.corpus = corpus
        self.hyde = hyde
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_freqs = self._calculate_doc_freqs()
        self.N = len(corpus)

    def _calculate_doc_freqs(self):
        """
        Calculate document frequencies for all terms in the corpus.
        """
        doc_freqs = Counter()
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        return doc_freqs

    def idf(self, term):
        """
        Calculate the IDF for a term.
        """
        n_t = self.doc_freqs.get(term, 0)
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query, doc):
        """
        Calculate the BM25 score for a query and a document.
        :param query: List of query terms.
        :param doc: List of document terms.
        :return: BM25 score.
        """
        score = 0
        doc_term_freqs = Counter(doc)
        for term in query:
            f_t_D = doc_term_freqs[term]
            idf_t = self.idf(term)
            numerator = f_t_D * (self.k1 + 1)
            denominator = f_t_D + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            score += idf_t * (numerator / denominator)
        return score

    def search(self, queries, top_k=10, expand=True):
        """
        Search the corpus for the top-k documents for each query.
        :param queries: List of queries, where each query is a list of terms.
        :param top_k: Number of top documents to return.
        :param expand: Whether to use HyDE query expansion.
        :return: List of top-k documents for each query.
        """
        results = []
        for query in queries:
            if expand:
                hypotheses = self.hyde._get_hypotheses(" ".join(query))
                query = list(set(query + [term for hypo in hypotheses for term in hypo.split()]))
            scores = [(i, self.score(query, doc)) for i, doc in enumerate(self.corpus)]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
            results.append(scores)
        return results