
import os, json, sys, argparse
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss, MarginMSELoss, MSELoss, CachedMultipleNegativesRankingLoss
from utils import *

def run_retrieval(model, queries, corpus, relevant_docs, outpath, batch_size):

    ndcg_at_k=[5, 10]

    if "qwen" in outpath.lower():
        ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="pd_dataset",
        batch_size=batch_size,
        query_prompt_name="query",
        ndcg_at_k=ndcg_at_k,
    )

    else:
        ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="pd_dataset",
        batch_size=batch_size,
        ndcg_at_k=ndcg_at_k,
    )
    results = ir_evaluator(model)
    return results

def run_evaluation_all(model, outpath, batch_size=64):
    all_metrics = {}

    queries, corpus, relevant_docs = load_pd_dataset()
    results = run_retrieval(model, queries, corpus, relevant_docs, outpath, batch_size)
    all_metrics["PD_dataset"] = results

    print ("Recall @ 5 --", all_metrics["PD_dataset"]["pd_dataset_cosine_recall@5"])

    queries, corpus, relevant_docs = load_devset_barexam()
    results = run_retrieval(model, queries, corpus, relevant_docs, outpath, batch_size)
    all_metrics["barexam"] = results

    with open(outpath + "-metrics.json", "w") as outfile:
        json.dump(all_metrics, outfile)

def main():
    os.environ["WANDB_DISABLED"] = "True"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="intfloat/e5-base-v2")
    parser.add_argument('--batch_size', type=int, default=64)        
    parser.add_argument('--only_eval', type=str, default="yes")        
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default="retrieval-results")
    parser.add_argument('--dataset_name', type=str, default="barexam_qa")
    parser.add_argument('--prompt_prefix', type=str, default="")
    parser.add_argument('--do_training', type=str, default="no")

    args = parser.parse_args()
    path = "retrieval-results"
    os.makedirs(path, exist_ok=True)

    seed = args.seed
    print (seed)


    if args.do_training == "yes":
        outpath = os.path.join(path, args.dataset_name + "_" + os.path.basename(args.model_name) + "-seed-" + str(seed))
    elif args.do_training == "no":
        outpath = os.path.join(path, os.path.basename(args.model_name))

    print (args.model_name)
    try:
        model = SentenceTransformer(args.model_name, model_kwargs={"torch_dtype": "bfloat16"})
    except:
        model = SentenceTransformer(args.model_name)        

    if hasattr(model, 'tokenizer'):
        model.tokenizer.model_input_names = ['input_ids', 'attention_mask']

    if args.do_training != "yes":
        run_evaluation_all(model, outpath, batch_size=args.batch_size)
        sys.exit(0)


    from peft import LoraConfig, TaskType

    if args.dataset_name == "synthetic_finetuned": # not supported, is based on proprietary briefs
        train = load_trainset_synthetic_optimized(outpath)
    elif args.dataset_name == "query_expansion": # not supported, is based on proprietary briefs
        train = load_trainset_synthetic_optimized_expanded(outpath) 
    elif args.dataset_name == "synthetic_naive": # not supported, is based on proprietary briefs
        train = load_trainset_synthetic_naive(outpath)
    elif args.dataset_name == "barexam_qa": # supported, as example
        train = load_trainset_barexam_qa(outpath)
    elif args.dataset_name == "lepard": # # not supported, but easy to replicate 
        train = load_trainset_lepard(outpath)

    if "Qwen" in args.model_name:
        model.max_seq_length = 1024
    else: # e5 and all-mpnet have seq length 512
        model.max_seq_length = 512

    loss = CachedMultipleNegativesRankingLoss(model)

    num_epochs = 3

    if "save_checkpoint" in args.model_name or "mpnet" in args.model_name:
        gradient_checkpointing = False # for these, gradient checkpointing doesn't work
    else:
        gradient_checkpointing = True

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=2e-5,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        save_strategy="no",
        save_total_limit=1,
        logging_steps=50,
        logging_first_step=True,
        run_name="retrieval-experiments",
        overwrite_output_dir=True,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed
    )    

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=loss,
    )

    print ("training starts")
    trainer.train()
    run_evaluation_all(model, outpath, batch_size=args.batch_size) 



if __name__ == "__main__":
    main()