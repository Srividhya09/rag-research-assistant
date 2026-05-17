import pandas as pd

df = pd.read_csv('evaluation/results.csv')

print(f"Total questions evaluated: {len(df)}")
print()
print("RAGAS EVALUATION SUMMARY")
print("="*40)
print(f"Faithfulness:     {df['faithfulness'].mean():.4f}")
print(f"Answer Relevancy: {df['answer_relevancy'].mean():.4f}")
print(f"Context Recall:   {df['context_recall'].mean():.4f}")
avg = (df['faithfulness'].mean() + df['answer_relevancy'].mean() + df['context_recall'].mean()) / 3
print(f"Overall Average:  {avg:.4f}")
print("="*40)
print()
print("Columns in CSV:", df.columns.tolist())
print()
print("Per-question breakdown:")
print(df.to_string())