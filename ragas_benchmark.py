"""
ragas_benchmark.py
==================
End-to-end RAGAS evaluation: runs your VectorRAG & GraphRAG pipelines live,
collects contexts + answers, then judges with Gemini 2.5 Pro.

Judge LLM : Google Gemini 2.5 Pro  (strongest available)
Embeddings: all-mpnet-base-v2  (same model used by both RAG pipelines)
RAGAS API : >= 0.2.x

Dependencies (add to existing venv):
  pip install ragas>=0.2.0 langchain-google-genai langchain-huggingface \
              sentence-transformers pandas matplotlib tqdm python-dotenv tabulate

Usage:
  python ragas_benchmark.py                          # uses built-in SQuAD samples
  python ragas_benchmark.py --squad-csv squad_train_dataset.csv --n-samples 50
  python ragas_benchmark.py --output-dir results/run1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# Local RAG pipelines
from vector_rag_search import VectorRAGSearch
from graph_service import GraphRAGProcessor

matplotlib.use("Agg")
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# ── Judge provider — change this one line ─────────────────────────────────────
# Options:
#   "gemini"         → Gemini 2.5 Flash via Google AI Studio (free, key already in .env)
#   "groq"           → Gemma 2 9B via Groq cloud (free tier, 30 RPM)
#   "ollama"         → llama3 locally — already installed on your machine (4.7GB)
#   "ollama_cloud"   → gpt-oss:20b-cloud via Ollama cloud (runs on their servers, not your PC)
JUDGE_PROVIDER = "ollama_cloud"

_JUDGE_MODELS = {
    "gemini":       "models/gemini-2.5-flash",
    "groq":         "gemma2-9b-it",
    "ollama":       "llama3:latest",
    "ollama_cloud": "gemma3:12b-cloud",    # Using the gemma3:12b-cloud you are running
}

@dataclass
class BenchmarkConfig:
    judge_model: str = field(default_factory=lambda: _JUDGE_MODELS[JUDGE_PROVIDER])
    judge_temperature: float = 0.0
    embedding_model: str = "all-mpnet-base-v2"
    vector_top_k: int = 5
    raise_on_failure: bool = False
    output_dir: str = "results"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.timestamp

CFG = BenchmarkConfig()

# Prompt identical to evaluate_squad_rag.py for fair comparison
UNIVERSAL_PROMPT = """\
You are a helpful conversational voice assistant.
Answer the [QUESTION] using ONLY the provided [CONTEXT].

Instructions:
1. Read the [CONTEXT] carefully.
2. Find the exact answer to the [QUESTION] in the [CONTEXT].
3. Provide a natural, conversational sentence as your answer.
4. If the answer is not in the [CONTEXT], say: "I'm sorry, I couldn't find that in the database."

[CONTEXT]
{context}

[QUESTION]
{question}
"""

# ─────────────────────────────────────────────────────────────────────────────
# LLM & Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def build_judge_llm():
    """
    Returns (raw_lc_llm, ragas_wrapped_llm).
    Change JUDGE_PROVIDER at the top of the file to switch backends.
    """
    if JUDGE_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        print(f"⚡  Judge LLM (Groq cloud — free): {CFG.judge_model}")
        lc_llm = ChatGroq(
            model=CFG.judge_model,
            temperature=CFG.judge_temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif JUDGE_PROVIDER in ("ollama", "ollama_cloud"):
        from langchain_ollama import ChatOllama
        label = "Ollama cloud" if JUDGE_PROVIDER == "ollama_cloud" else "local Ollama"
        print(f"🦙  Judge LLM ({label}): {CFG.judge_model}")
        lc_llm = ChatOllama(model=CFG.judge_model, temperature=CFG.judge_temperature)
    else:  # gemini (default)
        print(f"🤖  Judge LLM (Gemini API — free): {CFG.judge_model}")
        lc_llm = ChatGoogleGenerativeAI(
            model=CFG.judge_model,
            temperature=CFG.judge_temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return lc_llm, LangchainLLMWrapper(lc_llm)


def build_embeddings() -> LangchainEmbeddingsWrapper:
    print(f"📐  Embeddings: {CFG.embedding_model}")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=CFG.embedding_model))


def build_metrics(llm: LangchainLLMWrapper, emb: LangchainEmbeddingsWrapper) -> list[Any]:
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# SQuAD dataset loader
# ─────────────────────────────────────────────────────────────────────────────

# Fallback built-in samples if no CSV is provided
BUILTIN_SAMPLES = [
    {"question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
     "ground_truth": "Saint Bernadette Soubirous"},
    {"question": "What is in front of the Notre Dame Main Building?",
     "ground_truth": "a copper statue of Christ"},
    {"question": "When did the Scholastic Magazine of Notre dame begin publishing?",
     "ground_truth": "September 1876"},
    {"question": "The granting of Doctorate degrees first occurred in what year at Notre Dame?",
     "ground_truth": "1924"},
    {"question": "The Lobund Institute was merged into the Department of Biology at Notre Dame in what year?",
     "ground_truth": "1958"},
    {"question": "When did the first art gallery open in Washington state?",
     "ground_truth": "1927"},
    {"question": "On what type of transportation system has Seattle begun to focus?",
     "ground_truth": "mass transit"},
    {"question": "What can cause your memory to deterioriate or not work as well?",
     "ground_truth": "Stress"},
    {"question": "In studies what is a relationship between sleeping and learning?",
     "ground_truth": "activation patterns in the sleeping brain that mirror those recorded during the learning of tasks from the previous day"},
    {"question": "In the 1990s, what type of programming changed the handling of databases?",
     "ground_truth": "object-oriented"},
]


def load_squad_samples(csv_path: str | None, n_samples: int) -> list[dict]:
    """Load question/ground_truth pairs from SQuAD CSV or fall back to built-ins."""
    if csv_path and Path(csv_path).exists():
        print(f"📂  Loading SQuAD from: {csv_path}  (n={n_samples})")
        df = pd.read_csv(csv_path, nrows=n_samples)
        # SQuAD CSVs typically have 'question' and 'answers' columns
        # 'answers' may be a JSON string like {"text": ["answer"], "answer_start": [0]}
        samples = []
        for _, row in df.iterrows():
            gt = row.get("answers", row.get("ground_truth", ""))
            if isinstance(gt, str):
                try:
                    parsed = json.loads(gt)
                    gt = parsed["text"][0] if isinstance(parsed, dict) else gt
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
            samples.append({"question": str(row["question"]), "ground_truth": str(gt)})
        return samples[:n_samples]
    else:
        print(f"⚠️   No CSV found — using {len(BUILTIN_SAMPLES)} built-in SQuAD samples.")
        return BUILTIN_SAMPLES[:n_samples]

# ─────────────────────────────────────────────────────────────────────────────
# Latency tracking
# ─────────────────────────────────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000


def percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms":  float(np.percentile(arr, 50)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "max_ms":  float(np.max(arr)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline dataset builders
# ─────────────────────────────────────────────────────────────────────────────

async def build_vector_dataset(
    samples: list[dict],
    judge_llm: ChatGoogleGenerativeAI,
) -> pd.DataFrame:
    """
    For each sample, retrieve via VectorRAG then generate an answer with Gemini.
    Returns DataFrame with: question, contexts, answer, ground_truth, retrieval_ms, generation_ms
    """
    print("\n🔍  Building VectorRAG eval dataset...")
    vector_search = VectorRAGSearch()
    rows = []
    retrieval_times, gen_times = [], []

    for sample in tqdm(samples, desc="VectorRAG retrieval+generation"):
        q = sample["question"]
        gt = sample["ground_truth"]

        # Retrieval (sync, run in thread to not block event loop)
        with LatencyTracker() as ret_t:
            results = await asyncio.to_thread(vector_search.search, q, CFG.vector_top_k)
        retrieval_times.append(ret_t.elapsed_ms)

        contexts = [r["context"] for r in results]
        context_text = "\n".join(contexts)

        # Generation
        prompt = UNIVERSAL_PROMPT.format(context=context_text, question=q)
        with LatencyTracker() as gen_t:
            try:
                answer = (await asyncio.to_thread(
                    judge_llm.invoke, [HumanMessage(content=prompt)]
                )).content
            except Exception as e:
                print(f"  ⚠️  Generation failed for '{q[:40]}': {e}")
                answer = ""
        gen_times.append(gen_t.elapsed_ms)

        rows.append({
            "question": q,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": gt,
            "retrieval_ms": ret_t.elapsed_ms,
            "generation_ms": gen_t.elapsed_ms,
        })

    vector_search.close()
    print(f"    Retrieval  — {percentiles(retrieval_times)}")
    print(f"    Generation — {percentiles(gen_times)}")
    return pd.DataFrame(rows)


async def build_graph_dataset(
    samples: list[dict],
    judge_llm: ChatGoogleGenerativeAI,
) -> pd.DataFrame:
    """
    For each sample, retrieve via GraphRAG then generate an answer with Gemini.
    Returns DataFrame with: question, contexts, answer, ground_truth, retrieval_ms, generation_ms
    """
    print("\n🧠  Building GraphRAG eval dataset...")
    graph_processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
        neo4j_db=os.getenv("NEO4J_DB", "vectorchunkgraph"),
    )
    await graph_processor._initialize_graph()

    rows = []
    retrieval_times, gen_times = [], []

    for sample in tqdm(samples, desc="GraphRAG retrieval+generation"):
        q = sample["question"]
        gt = sample["ground_truth"]

        # Retrieval (async)
        with LatencyTracker() as ret_t:
            try:
                context_str = await graph_processor.retrieve(q)
            except Exception as e:
                print(f"  ⚠️  GraphRAG retrieval failed for '{q[:40]}': {e}")
                context_str = ""
        retrieval_times.append(ret_t.elapsed_ms)

        # Split the returned string back into chunks for RAGAS contexts list
        # GraphRAG returns a single formatted string — we treat each non-empty line as a chunk
        contexts = [ln for ln in context_str.splitlines() if ln.strip()]
        if not contexts:
            contexts = [context_str] if context_str else ["No context retrieved."]

        # Generation
        prompt = UNIVERSAL_PROMPT.format(context=context_str, question=q)
        with LatencyTracker() as gen_t:
            try:
                answer = (await asyncio.to_thread(
                    judge_llm.invoke, [HumanMessage(content=prompt)]
                )).content
            except Exception as e:
                print(f"  ⚠️  Generation failed for '{q[:40]}': {e}")
                answer = ""
        gen_times.append(gen_t.elapsed_ms)

        rows.append({
            "question": q,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": gt,
            "retrieval_ms": ret_t.elapsed_ms,
            "generation_ms": gen_t.elapsed_ms,
        })

    await graph_processor.driver.close()
    print(f"    Retrieval  — {percentiles(retrieval_times)}")
    print(f"    Generation — {percentiles(gen_times)}")
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluation engine
# ─────────────────────────────────────────────────────────────────────────────

RENAME = {
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_precision":"ContextPrecision",
    "context_recall":   "ContextRecall",
}


def df_to_ragas_dataset(df: pd.DataFrame) -> EvaluationDataset:
    samples = []
    for _, row in df.iterrows():
        ctxs = row["contexts"]
        if isinstance(ctxs, str):
            try:
                ctxs = json.loads(ctxs)
            except json.JSONDecodeError:
                ctxs = [ctxs]
        samples.append(SingleTurnSample(
            user_input=str(row["question"]),
            retrieved_contexts=ctxs,
            response=str(row["answer"]),
            reference=str(row["ground_truth"]),
        ))
    return EvaluationDataset(samples=samples)


async def run_ragas_eval(
    df: pd.DataFrame,
    pipeline_name: str,
    metrics: list[Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    print(f"\n{'='*60}")
    print(f"  ⚖️  RAGAS Evaluating: {pipeline_name}  ({len(df)} samples)")
    print(f"{'='*60}")

    dataset = df_to_ragas_dataset(df)
    loop = asyncio.get_event_loop()

    with tqdm(total=1, desc=f"RAGAS [{pipeline_name}]", unit="batch"):
        result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                dataset=dataset,
                metrics=metrics,
                raise_exceptions=CFG.raise_on_failure,
            ),
        )

    scores_df = result.to_pandas()
    meta = df[["question", "answer", "ground_truth", "retrieval_ms", "generation_ms"]].reset_index(drop=True)
    scores_df = pd.concat([meta, scores_df], axis=1)
    scores_df.rename(columns=RENAME, inplace=True)

    score_cols = [v for v in RENAME.values() if v in scores_df.columns]
    aggregate = {col: float(scores_df[col].mean()) for col in score_cols}

    print(f"\n✅  {pipeline_name} aggregate:")
    for k, v in aggregate.items():
        print(f"    {k:<22} {v:.4f}")

    return scores_df, aggregate

# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(vec_agg: dict, graph_agg: dict) -> pd.DataFrame:
    rows = []
    for metric in sorted(set(vec_agg) | set(graph_agg)):
        v = vec_agg.get(metric, float("nan"))
        g = graph_agg.get(metric, float("nan"))
        d = g - v
        arrow = "▲" if d > 0.005 else ("▼" if d < -0.005 else "─")
        rows.append({
            "Metric": metric,
            "VectorRAG": round(v, 4),
            "GraphRAG":  round(g, 4),
            "Δ (Graph−Vector)": round(d, 4),
            "Winner": f"{arrow} {'GraphRAG' if d > 0 else 'VectorRAG' if d < 0 else 'Tie'}",
        })
    cmp = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  📊  VectorRAG vs GraphRAG — Comparison")
    print("=" * 70)
    print(tabulate(cmp, headers="keys", tablefmt="fancy_grid", showindex=False))
    return cmp

# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_results(vec_df, graph_df, cmp_df, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    vec_df.to_csv(run_dir / "vectorrag_per_query_eval.csv", index=False)
    graph_df.to_csv(run_dir / "graphrag_per_query_eval.csv", index=False)
    cmp_df.to_csv(run_dir / "ragas_summary_comparison.csv", index=False)
    print(f"\n💾  Results saved to: {run_dir.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

METRICS   = ["Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall"]
C_VECTOR  = "#4C72B0"
C_GRAPH   = "#DD8452"


def plot_grouped_bar(vec_agg, graph_agg, run_dir):
    ms = [m for m in METRICS if m in vec_agg or m in graph_agg]
    x = np.arange(len(ms)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bv = ax.bar(x - w/2, [vec_agg.get(m, 0) for m in ms], w, label="VectorRAG", color=C_VECTOR, alpha=0.9)
    bg = ax.bar(x + w/2, [graph_agg.get(m, 0) for m in ms], w, label="GraphRAG",  color=C_GRAPH,  alpha=0.9)
    for bar in list(bv) + list(bg):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("VectorRAG vs GraphRAG — RAGAS Metrics", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(run_dir / "comparison_bar_chart.png", dpi=150); plt.close(fig)


def plot_radar(vec_agg, graph_agg, run_dir):
    ms = [m for m in METRICS if m in vec_agg or m in graph_agg]
    N = len(ms)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    v_vals = [vec_agg.get(m, 0) for m in ms] + [vec_agg.get(ms[0], 0)]
    g_vals = [graph_agg.get(m, 0) for m in ms] + [graph_agg.get(ms[0], 0)]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, v_vals, "o-", lw=2, label="VectorRAG", color=C_VECTOR)
    ax.fill(angles, v_vals, alpha=0.15, color=C_VECTOR)
    ax.plot(angles, g_vals, "o-", lw=2, label="GraphRAG",  color=C_GRAPH)
    ax.fill(angles, g_vals, alpha=0.15, color=C_GRAPH)
    ax.set_thetagrids(np.degrees(angles[:-1]), ms, fontsize=11)
    ax.set_ylim(0, 1); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("RAGAS Radar Chart", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout(); fig.savefig(run_dir / "radar_chart.png", dpi=150); plt.close(fig)


def plot_boxplots(vec_df, graph_df, run_dir):
    ms = [m for m in METRICS if m in vec_df.columns and m in graph_df.columns]
    fig, axes = plt.subplots(1, len(ms), figsize=(5*len(ms), 6))
    if len(ms) == 1: axes = [axes]
    for ax, m in zip(axes, ms):
        bp = ax.boxplot(
            [vec_df[m].dropna().tolist(), graph_df[m].dropna().tolist()],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="white", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(C_VECTOR)
        bp["boxes"][1].set_facecolor(C_GRAPH)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["VectorRAG", "GraphRAG"])
        ax.set_title(m, fontweight="bold"); ax.set_ylim(-0.05, 1.1)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-Sample Score Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(run_dir / "boxplot_distributions.png", dpi=150); plt.close(fig)


def plot_latency(vec_df, graph_df, run_dir):
    """Bar chart comparing mean retrieval + generation latency."""
    labels = ["VectorRAG", "GraphRAG"]
    ret_ms  = [vec_df["retrieval_ms"].mean(),  graph_df["retrieval_ms"].mean()]
    gen_ms  = [vec_df["generation_ms"].mean(), graph_df["generation_ms"].mean()]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, ret_ms, w, label="Retrieval",   color="#55a868", alpha=0.9)
    ax.bar(x + w/2, gen_ms, w, label="Generation",  color="#c44e52", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Latency (ms)"); ax.legend()
    ax.set_title("Mean Retrieval vs Generation Latency", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(run_dir / "latency_comparison.png", dpi=150); plt.close(fig)


def plot_all(vec_df, graph_df, vec_agg, graph_agg, run_dir):
    print("\n📈  Generating visualisations...")
    plot_grouped_bar(vec_agg, graph_agg, run_dir)
    plot_radar(vec_agg, graph_agg, run_dir)
    plot_boxplots(vec_df, graph_df, run_dir)
    plot_latency(vec_df, graph_df, run_dir)
    print(f"    Saved 4 charts to {run_dir.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if args.output_dir:
        CFG.output_dir = args.output_dir
    run_dir = CFG.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SQuAD questions + ground truths
    samples = load_squad_samples(args.squad_csv, args.n_samples)
    print(f"\n📋  Loaded {len(samples)} evaluation samples.")

    # 2. Init judge LLM + RAGAS wrappers
    judge_llm, ragas_llm = build_judge_llm()
    ragas_emb = build_embeddings()
    metrics = build_metrics(ragas_llm, ragas_emb)

    # 3. Build eval datasets by running live pipelines
    vec_df   = await build_vector_dataset(samples, judge_llm)
    graph_df = await build_graph_dataset(samples, judge_llm)

    # Optionally save raw pipeline outputs for inspection
    vec_df.to_csv(run_dir / "vectorrag_raw_outputs.csv", index=False)
    graph_df.to_csv(run_dir / "graphrag_raw_outputs.csv", index=False)

    # 4. Run RAGAS evaluation
    t0 = time.perf_counter()
    vec_scores_df,   vec_agg   = await run_ragas_eval(vec_df,   "VectorRAG", metrics)
    graph_scores_df, graph_agg = await run_ragas_eval(graph_df, "GraphRAG",  metrics)
    print(f"\n⏱️   Total RAGAS eval time: {time.perf_counter()-t0:.1f}s")

    # 5. Compare
    cmp_df = compare_results(vec_agg, graph_agg)

    # 6. Export
    export_results(vec_scores_df, graph_scores_df, cmp_df, run_dir)

    # 7. Visualise
    plot_all(vec_scores_df, graph_scores_df, vec_agg, graph_agg, run_dir)

    print(f"\n🎉  Benchmark complete! All outputs in: {run_dir.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS benchmark: VectorRAG vs GraphRAG")
    parser.add_argument("--squad-csv", type=str, default=None,
                        help="Path to SQuAD CSV (e.g. squad_train_dataset.csv). "
                             "Omit to use built-in 8 samples.")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of questions to evaluate (default: 8).")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Root output directory (default: results/).")
    args = parser.parse_args()
    asyncio.run(main(args))
