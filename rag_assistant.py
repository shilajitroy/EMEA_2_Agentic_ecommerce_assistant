"""Local Chroma FAQ retrieval with optional grounded LLM responses."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_config import LLMUnavailableError, get_chat_model, redact_sensitive_text


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
FAQ_PATH = PROJECT_DIR / "faq_data.csv"
CHROMA_PATH = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_faq"
NO_FAQ_ANSWER = "The available sample FAQ does not contain sufficient information to answer that."
MIN_RELEVANCE = 0.25


class FAQError(RuntimeError):
    """FAQ source or retrieval could not be used."""


def load_faq_documents(path: str | Path | None = None) -> list[Document]:
    """Convert FAQ CSV rows into LangChain documents with useful metadata."""
    faq_path = Path(path) if path else FAQ_PATH
    if not faq_path.is_file():
        raise FAQError("The FAQ file faq_data.csv was not found.")
    try:
        records = pd.read_csv(faq_path).fillna("")
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise FAQError("The FAQ file could not be read.") from error
    if not {"question", "answer"}.issubset(records.columns):
        raise FAQError("The FAQ file must contain question and answer columns.")

    documents = []
    for index, row in records.iterrows():
        question, answer = str(row["question"]).strip(), str(row["answer"]).strip()
        if not question or not answer:
            continue
        documents.append(
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={
                    "question": question,
                    "topic": str(row.get("topic", "general")),
                    "faq_id": str(index),
                },
            )
        )
    if not documents:
        raise FAQError("The FAQ file contains no usable question-and-answer rows.")
    return documents


@lru_cache(maxsize=1)
def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    load_dotenv(PROJECT_DIR / ".env")
    model_name = os.getenv(
        "FAQ_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Open the persistent FAQ store and populate it once when empty."""
    try:
        from langchain_chroma import Chroma

        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_PATH),
            embedding_function=_get_embeddings(),
        )
        if not vector_store.get().get("ids"):
            vector_store.add_documents(load_faq_documents())
        return vector_store
    except FAQError:
        raise
    except Exception as error:
        LOGGER.exception("Could not initialize the local FAQ vector database.")
        raise FAQError(
            "FAQ search is not available. Check the FAQ file and local embedding setup."
        ) from error


def retrieve_faq(question: str, limit: int = 4) -> list[dict[str, object]]:
    """Return relevant FAQ entries, filtering away weak semantic matches."""
    if not isinstance(question, str) or not question.strip():
        raise FAQError("Enter a question to search the FAQ.")
    try:
        results = get_vector_store().similarity_search_with_relevance_scores(
            question.strip(),
            k=max(1, min(limit, 8)),
        )
    except FAQError:
        raise
    except Exception as error:
        LOGGER.exception("FAQ vector search failed.")
        raise FAQError("FAQ search failed. Please try again.") from error

    retrieved = []
    for document, score in results:
        if score < MIN_RELEVANCE:
            continue
        retrieved.append(
            {
                "question": str(document.metadata.get("question", "")),
                "answer": document.page_content.split("\nAnswer: ", 1)[-1],
                "topic": str(document.metadata.get("topic", "general")),
                "relevance": round(float(score), 3),
            }
        )
    return retrieved


def answer_faq(question: str) -> dict[str, object]:
    """Answer from FAQ retrieval; only use a configured LLM to paraphrase evidence."""
    matches = retrieve_faq(question)
    if not matches:
        return {
            "answer": NO_FAQ_ANSWER,
            "sources": [],
            "mode": "retrieved-faq",
        }

    context = "\n\n".join(
        f"FAQ question: {entry['question']}\nFAQ answer: {entry['answer']}"
        for entry in matches
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer using only the supplied FAQ entries. If they do not "
                    "answer the question, say the available sample FAQ does not "
                    "contain sufficient information. Never invent warranty, "
                    "returns, prices, or policies.",
                ),
                ("human", "FAQ entries:\n{context}\n\nQuestion: {question}"),
            ]
        )
        answer = (prompt | get_chat_model() | StrOutputParser()).invoke(
            {
                "context": context,
                "question": redact_sensitive_text(question.strip()),
            }
        ).strip()
        if answer:
            mode = "llm-grounded"
        else:
            answer = str(matches[0]["answer"])
            mode = "retrieved-faq"
    except LLMUnavailableError:
        answer = "\n\n".join(str(entry["answer"]) for entry in matches[:2])
        mode = "retrieved-faq"
    except Exception as error:
        LOGGER.warning("Optional FAQ LLM failed; using retrieved FAQ text: %s", type(error).__name__)
        answer = "\n\n".join(str(entry["answer"]) for entry in matches[:2])
        mode = "retrieved-faq"

    return {
        "answer": answer,
        "sources": [
            {"question": entry["question"], "topic": entry["topic"]}
            for entry in matches
        ],
        "mode": mode,
    }
