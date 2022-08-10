import io
import json
import pickle
from typing import List, Any, Dict

from tqdm import tqdm
import torch
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile

OUTPUT_PATH = "gs://cohere-dev/eve/data/retrieval/beir/nq-train/nq-train_cpt-small_top100.json"

QUERY_EMBEDDING_PATH = "gs://cohere-dev/eve/data/retrieval/beir/nq-train/embeddings/query_embeddings_train.npy"
QUERY_INDEX_PATH = "gs://cohere-dev/eve/data/retrieval/beir/nq-train/embeddings/query_id.txt"
CORPUS_INDEX_PATH = "gs://cohere-dev/eve/data/retrieval/beir/nq-train/embeddings/sorted_corpus_id.txt"
corpus_embedding_path_fn = lambda n_chunk: f"gs://cohere-dev/eve/data/retrieval/beir/nq-train/embeddings/corpus_embeddings_chunk{n_chunk}.npy"


# util function from tif :))
def read_npy(path: str) -> List[Any]:
    with tf.io.gfile.GFile(path, 'rb') as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        loaded = np.load(f_io)

    return loaded


def read_txt(path: str) -> List[Any]:
    with tf.io.gfile.GFile(path, "r") as f:
        txt = f.read()
    txt_list = eval(txt)
    return txt_list


def compute_sim_by_chunk(k: int=100, normalize: bool=True):
    query_embeddings = read_npy(QUERY_EMBEDDING_PATH)
    query_ids = read_txt(QUERY_INDEX_PATH)
    corpus_ids = read_txt(CORPUS_INDEX_PATH)
    n_chunks = len(gfile.glob(corpus_embedding_path_fn("*")))
    result = {qid: {} for qid in query_ids}
    corpus_start_idx = 0
    if normalize:
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings)
    for i in tqdm(range(1, n_chunks+1), total=n_chunks):
        print(f"Computing similarities for chunk # {i} ...")
        sub_corpus_embeddings = read_npy(corpus_embedding_path_fn(str(i)))
        if normalize:
            sub_corpus_embeddings = sub_corpus_embeddings / np.linalg.norm(sub_corpus_embeddings)
        corpus_end_idx = corpus_start_idx + len(sub_corpus_embeddings)
        sub_corpus_ids = corpus_ids[corpus_start_idx:corpus_end_idx]
        similarity_scores = query_embeddings @ sub_corpus_embeddings.T
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(torch.tensor(similarity_scores), min(k + 1, len(similarity_scores[1])), dim=1, largest=True, sorted=False)
        cos_scores_top_k_values= cos_scores_top_k_values.detach().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.detach().tolist()
        for qid_idx, qid in enumerate(query_ids):
            top_k_ids = np.array(sub_corpus_ids)[cos_scores_top_k_idx[qid_idx]].tolist()
            for cid, cossim in zip(top_k_ids, cos_scores_top_k_values[qid_idx]):
                result[qid][cid] = cossim
        corpus_start_idx = corpus_end_idx
    return result


def get_top_k_retrieved_results(qid_cid_score: Dict[int, Dict[int, float]], k: int=100):
    new_result = {}
    for qid in qid_cid_score:
        top_k_idx = sorted(qid_cid_score[qid].items(), key=lambda item: item[1], reverse=True).keys().tolist()[1:k+1]
        new_result[qid] = top_k_idx

    with gfile.GFile(OUTPUT_PATH, "w") as f:
        json.dump(new_result, f)
    
    print(f"Wrote top 100 hits to {OUTPUT_PATH} 🎉")


if __name__ == "__main__":
    result = compute_sim_by_chunk()
    get_top_k_retrieved_results(qid_cid_score=result)
