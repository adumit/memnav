from transformers import pipeline
import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from nltk import sent_tokenize, download
from itertools import chain


def _test_nltk_download():
    try:
        sent_tokenize("abc")
    except:
        print("Could not run sent_tokenize... downloading the punkt model now.")
        download("punkt")

# Run this on startup to ensure the rest of the script will work
_test_nltk_download()


def get_all_text_files(root_dir):
    expanded_file_locations = []
    for dirpath, dirs, files in os.walk(root_dir):
        for fname in files:
            if ".md" or ".txt" in fname:
                expanded_file_locations.append(f"{dirpath}/{fname}")
    return expanded_file_locations

class MemNav:
    def __init__(self, root_dir='.'):
        """Load models, preprocess text, precompute embeddings."""
        self.root_dir = root_dir

        # Load language models
        self.qa = pipeline('question-answering')
        self.sum = pipeline('summarization')
        self.text_encoder = SentenceTransformer('msmarco-distilbert-base-v2')
        self.pair_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

        # Load list of entries
        md_files = get_all_text_files(self.root_dir)
        self.entries = [open(fpath).read() for fpath in md_files]

        # Tokenize entries into sentences
        self.entries = [sent_tokenize(entry.strip()) for entry in self.entries]

        # Merge each 3 consecutive sentences into one passage
        self.entries = list(chain(*[[' '.join(entry[start_idx:min(start_idx + 3, len(entry))])
                                     for start_idx in range(0, len(entry), 3)] for entry in self.entries]))

        # Pre-compute passage embeddings
        self.passage_embeddings = self.text_encoder.encode(
            self.entries, show_progress_bar=True)

    def retrieval(self, query):
        """Utility for retrieving passages most relevant to a given query."""
        # First pass, find passages most similar to query
        question_embedding = self.text_encoder.encode(
            query, convert_to_tensor=True)
        hits = util.semantic_search(
            question_embedding, self.passage_embeddings, top_k=100)[0]

        # Second pass, re-rank passages more thoroughly
        cross_scores = self.pair_encoder.predict(
            [[query, self.entries[hit['corpus_id']]] for hit in hits])

        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        # Select best few results
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

        results = []
        for hit in hits[:5]:
            if hit['cross-score'] > 1e-3:
                results += [self.entries[hit['corpus_id']]]

        return results

    def search(self, query):
        """Search knowledge base for passages most relevant to a given query."""
        print(*self.retrieval(query), sep='\n\n')

    def ask(self, question):
        """Obtain an answer to a question posed to the knowledge base. Provides retrieved passages as context for a question-answering pipeline."""
        return self.qa(question, ' '.join(self.retrieval(question)))['answer']

    def summarize(self, query):
        """Obtain a summary related to the query using the knowledge base. Provides retrieved passages as input for a summarization pipeline."""
        return self.sum(' '.join(self.retrieval(query)), 130, 30, False)[0]['summary_text']
