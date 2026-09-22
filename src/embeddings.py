from sentence_transformers import SentenceTransformer
from chunking import create_chunks_from_pages
from ingestion import extract_text_from_pdf
from pathlib import Path
import json
import numpy as np


DATA_DIR = Path("data")
INDEX_DIR = Path("index")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_all_chunks():
    """
    Extrait les documents et crée les chunks
    avec leurs métadonnées.
    """

    all_chunks = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:

        print(f"\nTraitement : {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)

        document_chunks = create_chunks_from_pages(pages)

        all_chunks.extend(document_chunks)

        print(
            f"  → {len(document_chunks)} chunks générés"
        )

    return all_chunks


def main():

    INDEX_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------
    # 1. Chargement du modèle
    # --------------------------------------------------

    print("=" * 80)
    print("GÉNÉRATION DES EMBEDDINGS")
    print("=" * 80)

    print("\nChargement du modèle d'embedding...")

    model = SentenceTransformer(MODEL_NAME)

    print("Modèle chargé.")


    # --------------------------------------------------
    # 2. Création des chunks
    # --------------------------------------------------

    print("\nCréation des chunks...")

    chunks = load_all_chunks()

    print(f"\nNombre total de chunks : {len(chunks)}")


    # --------------------------------------------------
    # 3. Sauvegarde des chunks
    # --------------------------------------------------

    chunks_path = INDEX_DIR / "chunks.json"

    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Chunks sauvegardés dans : {chunks_path}")


    # --------------------------------------------------
    # 4. Préparation des textes
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # --------------------------------------------------
    # 5. Génération des embeddings
    # --------------------------------------------------

    print("\nGénération des embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Embeddings générés.")


    # --------------------------------------------------
    # 6. Sauvegarde des embeddings
    # --------------------------------------------------

    embeddings_path = INDEX_DIR / "embeddings.npy"

    np.save(
        embeddings_path,
        embeddings
    )

    print(
        f"Embeddings sauvegardés dans : "
        f"{embeddings_path}"
    )


    # --------------------------------------------------
    # 7. Informations finales
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    print(f"Nombre de chunks : {len(chunks)}")
    print(f"Nombre de vecteurs : {len(embeddings)}")
    print(f"Dimension : {embeddings.shape[1]}")


if __name__ == "__main__":
    main()