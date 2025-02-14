# src/rag_skeleton/batch_rag.py
import argparse
import re
from pathlib import Path

import pandas as pd

from bioalloyrag.rag import RAGPipeline

# Auto-detect `vectordb_path` based on `run.py`
DEFAULT_VECTORDB_PATH = Path(__file__).resolve().parent / "data" / "vectordb"


def process_questions(
    csv_path,
    output_path,
    vectordb_path=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    load_mode="local",
    api_token=None,
):
    """
    Reads a CSV file with a 'question' column, processes each question using the RAG pipeline,
    and saves responses in a new column named 'answer'.
    """

    # If vectordb_path is not provided, use the correct default
    vectordb_path = vectordb_path or DEFAULT_VECTORDB_PATH
    vectordb_path = str(Path(vectordb_path).resolve())  # Convert to absolute path

    print(f"Using vector database at: {vectordb_path}")

    # Ensure FAISS index exists
    if not Path(vectordb_path, "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {vectordb_path}. Run the RAG pipeline first."
        )

    # Load CSV
    df = pd.read_csv(csv_path)

    if "question" not in df.columns:
        raise ValueError("The CSV file must have a 'question' column.")

    # Initialize the RAG pipeline
    rag_pipeline = RAGPipeline(
        vectordb_path=vectordb_path,
        model_name=model_name,
        load_mode=load_mode,
        api_token=api_token,
        use_history=False,
    )
    rag_pipeline.setup_pipeline()

    # Function to get both answer and retrieved context
    def process_query(question):
        retrieved_docs = rag_pipeline.retriever.get_retriever().invoke(question)

        # ✅ Format retrieved contexts as a list with newline-separated items
        formatted_context = (
            "[\n"
            + ",\n".join(f"'{doc.page_content}'" for doc in retrieved_docs)
            + "\n]"
        )

        response = rag_pipeline.get_response(question)

        # ✅ Remove "For further reference, look at:" and everything after it
        cleaned_response = re.split(
            r"\n\nFor further reference, look at:", response, maxsplit=1
        )[0]

        return (
            cleaned_response.strip(),
            formatted_context,
        )  # ✅ Correctly formatted contexts list

    # Apply processing to each question
    df[["answer", "contexts"]] = df["question"].apply(
        lambda q: pd.Series(process_query(q))
    )

    # Save the updated CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process questions from a CSV file using the RAG pipeline."
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "output_path", type=str, help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name.",
    )
    parser.add_argument(
        "--load_mode",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Load model locally or via API.",
    )
    parser.add_argument(
        "--api_token",
        type=str,
        default=None,
        help="Hugging Face API token if using API mode.",
    )

    args = parser.parse_args()

    process_questions(
        args.csv_path,
        args.output_path,
        model_name=args.model_name,
        load_mode=args.load_mode,
        api_token=args.api_token,
    )
