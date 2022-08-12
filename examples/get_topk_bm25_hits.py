from itertools import chain

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
import pathlib, os, tqdm
import logging
import json
from tensorflow.io import gfile
from multiprocessing.pool import ThreadPool

NUM_THREADS = 2 * os.cpu_count()  # 2 threads per cpu core is standard


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "trec-news"
split = "test"
out_path = f"gs://cohere-dev/eve/data/retrieval/beir/{dataset}"
writing_freq = 512

# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
url = "http://202.61.230.171/beir/robust04.zip"
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# SciFact is a relatively small dataset! (limit shards to 1)
# number_of_shards = 1
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

hostname = "https://localhost:9200"
# (2) For datasets with big corpus ==> keep default configuration
model = BM25(index_name=dataset, hostname=hostname, initialize=initialize, timeout=10000)
bm25 = EvaluateRetrieval(model)

#### Index passages into the index (seperately) - takes a bit more than half an hour for msmarco
bm25.retriever.index(corpus)

triplets = []
retrieved_qrels = {}
iter = 0
batch_size = 64
sub_batch_size = 16
qids = list(qrels)
hard_negatives_max = 100

# if gfile.exists(os.path.join(out_path, f"{dataset}_bm25_top100.json")):
#     with gfile.GFile(os.path.join(out_path, f"{dataset}_bm25_top100.json"), "r+") as f:
#         retrieved_qrels = json.load(f)
#         starting_idx = len(retrieved_qrels)
#         print(f"Loaded cache for {dataset} at index {starting_idx-1}.")
# else:
starting_idx = 0

#### Retrieve BM25 hard negatives => Given a positive document, find most similar lexical documents
with ThreadPool(NUM_THREADS) as pool:
    for idx in tqdm.tqdm(range(starting_idx, len(qids), batch_size), desc="Retrieve Hard Negatives using BM25"):
        # If we are near the end don't go over
        new_idx = idx + batch_size if idx + batch_size < len(qids) else len(qids) - 1
        query_id = [qids[i] for i in range(idx, new_idx)] # (batch_size, )
        query_text = [queries[qid] for qid in query_id] # (batch_size, )

        query_text_nested = [query_text[i:i+sub_batch_size] for i in range(0, len(query_text), sub_batch_size)]

        ##### Multiprocessing
        search_fn = bm25.retriever.es.lexical_multisearch
        # 3 input args: list of texts, number of hits, and n_skip (skip to top n hits if set to n)
        inputs = [(query_list, hard_negatives_max + 1, 0) for query_list in query_text_nested]
        hits = list(pool.imap(search_fn, inputs))
        hits_unnested = list(chain(*hits))

        # hits = bm25.retriever.es.lexical_multisearch(texts=query_text, top_hits=hard_negatives_max)

        # (sylvie): hits is a list of dictionaries with two keys: "meta" and "hits". len(hits) = len(pos_docs). 
        # hits[k]["hits"] is a list of tuples with passage index and their relevance scores to the positive doc
        temp_retrieved = {}
        for qid, hit in zip(query_id, hits_unnested):
            if len(hit["hits"]) < hard_negatives_max:
                print(f"query {qid} only has {len(hit['hits'])} hits - skipping...")
                continue
            for i in range(hard_negatives_max):
                temp_retrieved[qid] = {hit["hits"][i][0]: hit["hits"][i][1]}

        retrieved_qrels.update(temp_retrieved)
        iter += batch_size

        if iter % writing_freq == 0 or iter == batch_size:
            with gfile.GFile(os.path.join(out_path, f"{dataset}_{split}_bm25_top100.json"), "w") as f:
                json.dump(retrieved_qrels, f)
                
    with gfile.GFile(os.path.join(out_path, f"{dataset}_{split}_bm25_top100.json"), "w") as f:
        json.dump(retrieved_qrels, f)
        print(f"Wrote bm25 top {hard_negatives_max} hits for {dataset} 🎉")

