import pandas as pd
import os, json
from collections import defaultdict
from datasets import load_dataset, Dataset

def load_trainset_barexam_qa(outpath):
    questions = load_dataset("barexamqa-mbe", "qa")

    gold, qs, gold_paragraphs = [], [], []
    for i, row in enumerate(questions["train"]):
        q = row["prompt"] + " " + row["question"]
        qs.append(q.strip().lower())
        gold.append(row["gold_idx"])
        gold_paragraphs.append(row["gold_passage"].strip())

    if "Qwen" in outpath or "mistral" in outpath:
        prompt_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        qs = [prompt_prefix + i for i in qs]

    train = Dataset.from_dict({"question": qs, "answer": gold_paragraphs})
    return train


def load_devset_barexam():
    questions = load_dataset("barexamqa-mbe", "qa")
    passages = load_dataset("barexamqa-mbe", "passages")

    qs, gold_paragraphs = [], []
    for i, row in enumerate(questions["validation"]):
        q = row["prompt"] + " " + row["question"]
        qs.append(q.strip().lower())
        gold_paragraphs.append(" ".join(row["gold_passage"].split()).strip())

    corpus_text = passages["validation"]["text"]
    corpus_text = [" ".join(i.split()).strip() for i in corpus_text]

    corpus_ids = list(range(len(corpus_text)))
    corpus = dict(zip(corpus_ids, corpus_text))  # Our corpus (cid => document)
    paragraph2idx = {j:i for i,j in corpus.items()}

    query_ids = list(range(len(qs)))
    queries = dict(zip(query_ids, qs))
    queries2id = {i:j for j,i in queries.items()}
    relevant_docs = defaultdict(set)

    for query, paragraph in zip(qs, gold_paragraphs):
        qid = queries2id[query]
        corpus_id = paragraph2idx[paragraph]
        relevant_docs[qid].add(corpus_id)

    return queries, corpus, relevant_docs


def load_pd_dataset():
    data_dir = "data"
    test = pd.read_csv(os.path.join(data_dir, "queries_and_targets.csv"))
    corpus_rows = [json.loads(line) for line in open(os.path.join(data_dir, "corpus.jsonl"))]

    queries = {row["query_idx"]: row["query"] for _, row in test.iterrows()}
    corpus = {int(r["id"]): r["contents"] for r in corpus_rows}

    relevant_docs = defaultdict(set)
    for _, row in test.iterrows():
        relevant_docs[row["query_idx"]].add(row["gold_idx"])
    return queries, corpus, relevant_docs
