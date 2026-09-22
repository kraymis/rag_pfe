"""Streamlit chat interface for the existing PFE RAG pipeline."""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def _ensure_project_imports() -> None:
    """Make the existing modules importable when Streamlit runs this file."""
    source_dir = str(Path(__file__).resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)


@st.cache_resource(show_spinner="Chargement des ressources RAG...")
def load_rag_resources():
    """Load the embedding model, FAISS index, and chunks once per app session."""
    _ensure_project_imports()
    from retrieval import load_resources

    return load_resources()


def _friendly_error(error: Exception) -> str:
    """Return a short, user-facing message without exposing implementation details."""
    message = str(error).lower()

    if "nvidia_api_key" in message:
        return "La clé NVIDIA_API_KEY est manquante. Ajoutez-la dans le fichier .env."
    if "faiss" in message or "index" in message:
        return "L'index FAISS est introuvable. Préparez l'index avant de poser une question."
    if "chunks" in message:
        return "Le fichier des chunks est introuvable. Préparez les données avant de poser une question."
    if "503" in message or "service unavailable" in message:
        return "Le service NVIDIA est temporairement indisponible. Réessayez dans quelques instants."
    if "429" in message or "rate limit" in message:
        return "Le service NVIDIA est momentanément limité. Réessayez dans quelques instants."
    if "timed out" in message or "timeout" in message:
        return "Le service NVIDIA met trop de temps à répondre. Réessayez dans quelques instants."
    if "connection" in message or "connect" in message:
        return "La connexion au service NVIDIA a échoué. Vérifiez votre connexion puis réessayez."
    if "401" in message or "403" in message or "authentication" in message:
        return "La clé NVIDIA_API_KEY est invalide ou non autorisée."
    return "Une erreur est survenue pendant le traitement de la question."


def answer_question(question: str) -> tuple[str, list[dict]]:
    """Run retrieval and generation using the project's existing functions."""
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError(
            "La variable NVIDIA_API_KEY est absente. Ajoutez-la dans le fichier .env."
        )

    if not FAISS_INDEX_PATH.is_file():
        raise FileNotFoundError(
            "L'index FAISS est introuvable. Lancez d'abord la construction de l'index."
        )
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            "Le fichier des chunks est introuvable. Lancez d'abord la préparation des données."
        )

    _ensure_project_imports()
    from generation import generate_answer
    from retrieval import search_mmr

    previous_directory = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        model, index, chunks = load_rag_resources()
        results = search_mmr(
            question,
            model,
            index,
            chunks,
            top_k=5,
            candidate_k=15,
            lambda_param=0.7,
        )
        answer = generate_answer(question, results)
    finally:
        os.chdir(previous_directory)

    return answer, results


def _source_label(metadata: dict) -> str:
    reference = metadata.get("reference", metadata.get("source", "Document"))
    company = metadata.get("entreprise", "Entreprise inconnue")
    year = metadata.get("annee", "Année inconnue")
    return f"{reference} — {company} — {year}"


def _render_sources(results: list[dict]) -> None:
    if not results:
        return

    with st.expander(f"Sources récupérées ({len(results)})", expanded=True):
        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            page = metadata.get("page", "?")
            section = metadata.get("section", "Section non précisée")
            st.markdown(f"**Source {rank}**")
            st.markdown(f"**{_source_label(metadata)}**")
            st.caption(f"Page {page} · {section}")
            with st.expander("Afficher le passage"):
                st.write(result.get("text", ""))


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))


def main() -> None:
    st.set_page_config(
        page_title="PFE RAG Assistant",
        page_icon="💬",
        layout="centered",
    )
    st.title("PFE RAG Assistant")
    st.caption(
        "Posez une question sur les quatre sujets de PFE. "
        "Les réponses sont fondées uniquement sur les documents fournis."
    )

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        _render_message(message)

    question = st.chat_input("Posez votre question...")
    if not question or not question.strip():
        return

    question = question.strip()
    messages.append({"role": "user", "content": question})
    _render_message(messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents et génération de la réponse..."):
            try:
                answer, results = answer_question(question)
            except Exception as error:
                user_message = _friendly_error(error)
                st.error(user_message)
                messages.append(
                    {
                        "role": "assistant",
                        "content": user_message,
                        "sources": [],
                    }
                )
                return

        st.markdown(answer)
        _render_sources(results)

    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results,
        }
    )


if __name__ == "__main__":
    main()
