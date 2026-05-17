import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.chain import build_chain

load_dotenv()

EVAL_PATH = Path("evaluation/eval_dataset.json")
RESULTS_PATH = Path("evaluation/results_v2.csv")


def load_eval_dataset():
    with open(EVAL_PATH, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} evaluation questions")
    return data


def run_rag_on_dataset(chain, eval_data):
    results = []
    for i, item in enumerate(eval_data):
        print(f"Running question {i+1}/{len(eval_data)}: {item['question'][:50]}...")
        try:
            output = chain.invoke({"query": item["question"]})
            answer = output["result"]
            source_docs = output["source_documents"]
            contexts = [doc.page_content for doc in source_docs]
            results.append({
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"]
            })
            time.sleep(1)
        except Exception as e:
            print(f"Error on question {i+1}: {e}")
            results.append({
                "question": item["question"],
                "answer": "Error",
                "contexts": [],
                "ground_truth": item["ground_truth"]
            })
    return results


def run_ragas_evaluation(results):
    print("\nRunning RAGAS evaluation in batches to avoid rate limits...")

    all_scores = []
    batch_size = 5

    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(results) + batch_size - 1) // batch_size
        print(f"\nEvaluating batch {batch_num}/{total_batches} "
              f"(questions {i+1}-{min(i+batch_size, len(results))})...")

        dataset = Dataset.from_list(batch)

        groq_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        ragas_llm = LangchainLLMWrapper(groq_llm)

        hf_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

        try:
            scores = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False
            )
            batch_df = scores.to_pandas()
            all_scores.append(batch_df)
            print(f"Batch {batch_num} done.")

            if i + batch_size < len(results):
                print("Waiting 30 seconds before next batch...")
                time.sleep(30)

        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
            print("Waiting 30 seconds before next batch...")
            time.sleep(30)

    final_df = pd.concat(all_scores, ignore_index=True)
    return final_df


def save_results(df):
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nDetailed results saved to {RESULTS_PATH}")
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION SUMMARY — FINAL")
    print("=" * 50)
    for col in ['faithfulness', 'answer_relevancy', 'context_recall']:
        if col in df.columns:
            valid = df[col].dropna()
            print(f"{col:20s}: {valid.mean():.4f} "
                  f"(from {len(valid)}/25 questions)")
    avg_cols = []
    for col in ['faithfulness', 'answer_relevancy', 'context_recall']:
        if col in df.columns:
            avg_cols.append(df[col].dropna().mean())
    if avg_cols:
        print(f"{'Overall Average':20s}: {sum(avg_cols)/len(avg_cols):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    print("Building RAG chain...")
    chain = build_chain(k=8)

    print("\nLoading evaluation dataset...")
    eval_data = load_eval_dataset()

    print("\nRunning RAG on all questions...")
    results = run_rag_on_dataset(chain, eval_data)

    print("\nStarting RAGAS evaluation...")
    df = run_ragas_evaluation(results)

    save_results(df)