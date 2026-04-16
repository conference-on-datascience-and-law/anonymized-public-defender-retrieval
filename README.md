# Public Defender Retrieval

AI tools are increasingly suggested as solutions to assist public agencies with heavy workloads. In public defense, where a constitutional right to counsel meets the complexities of law, overwhelming caseloads and constrained resources, practitioners face especially taxing conditions. Yet, there is little evidence of how AI could meaningfully support defenders' day-to-day work. In partnership with a public defender office, we developed a retrieval tool which surfaces relevant briefs to streamline legal research and writing.

We show that existing legal retrieval benchmarks fail to transfer to public defense search, however adding domain knowledge improves retrieval quality. This includes query expansion with legal reasoning, domain-specific data and curated synthetic examples. To facilitate further research, we provide a taxonomy of realistic defender search queries and release a manually annotated public defense retrieval dataset.
Together, our work offers starting points towards building practical, reliable retrieval AI tools for public defense, and towards more realistic legal retrieval benchmarks.

This repo contains 
- anonymized version of the dataset (anonymized with gemma4-31B)
- minimal reproduction code

```python
from datasets import load_dataset
#TODO
```

To run evaluation with [SentenceTransformers](https://sbert.net/)

```
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer

from collections import defaultdict

queries = {i:j for i,j in zip(test["query_idx"], test["query"])}
corpus = {int(i):j for i,j in zip(corpus["id"], corpus["contents"])}    

relevant_docs = defaultdict(set)
for example in test:
  relevant_docs[example["query_idx"]].add(example["gold_idx"])

model = SentenceTransformer("intfloat/e5-base-v2")        

ir_evaluator = InformationRetrievalEvaluator(
queries=queries,
corpus=corpus,
relevant_docs=relevant_docs) 

results = ir_evaluator(model)
```

To reproduce retrieval experiments, install [SentenceTransformers](https://sbert.net/)

## Zero-shot experiments
```python
python src/retrieval_experiments.py --do_training no --model_name intfloat/e5-base-v2
# Metrics are stored in folder retrieval-results
```

## Fine-tuning experiments
```python
# create custom data loaders in file src/utils.py for other datasets apart from barexam_qa
python src/retrieval_experiments.py --do_training yes --model_name intfloat/e5-base-v2 --dataset barexam_qa
# Metrics are stored in folder retrieval-results
```


