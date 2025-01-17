from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# dataset = "trec-covid"

# #### Download nfcorpus.zip dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

corpus_chunk_size = 50000
model_name = "/home/sylvie_cohere_ai/beir/models/make-multilingual-sys-2022-09-21_03-23-15_epoch0"
model = DRES(models.SentenceBERT(model_name), batch_size=256, corpus_chunk_size=corpus_chunk_size)
retriever = EvaluateRetrieval(model, score_function="dot")

# skipping languages with >5m texts in corpus ('english', 'japanese', 'russian')
languages = ['arabic', 'bengali', 'finnish', 'indonesian', 'korean', 'swahili', 'telugu', 'thai']
for lang in languages:
    print(f"LANGUAGE: {lang}!!!")
    data_path = f"examples/retrieval/evaluation/dense/datasets/mrtydi/{lang}"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

# recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
# hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

#### Print top-k documents retrieved ####
# top_k = 10

# query_id, ranking_scores = random.choice(list(results.items()))
# scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
# logging.info("Query : %s\n" % queries[query_id])

# for rank in range(top_k):
#     doc_id = scores_sorted[rank][0]
#     # Format: Rank x: ID [Title] Body
#     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))