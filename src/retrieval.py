import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_DIR = Path("index")

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 5
CANDIDATE_K = 15
LAMBDA = 0.7


# ============================================================
# CHARGEMENT DES RESSOURCES
# ============================================================



def load_resources():
    """
    Charge :
    - le modèle d'embedding
    - l'index FAISS
    - les chunks et leurs métadonnées
    """

    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return model, index, chunks


# ============================================================
# RECHERCHE PAR SIMILARITÉ
# ============================================================

def search_similarity(query, model, index, chunks, top_k=TOP_K):
    """
    Recherche classique :
    retourne les chunks les plus proches de la question
    selon la distance FAISS.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):
        chunk = chunks[index_position]

        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "distance": float(distance),
            "index": int(index_position)
        })

    return results


# ============================================================
# MMR
# ============================================================

def search_mmr(
    query,
    model,
    index,
    chunks,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
    lambda_param=LAMBDA
):
    """
    Recherche avec MMR + diversification par document.

    Le score prend en compte :
    - la pertinence du chunk par rapport à la question ;
    - la redondance avec les chunks déjà sélectionnés ;
    - la diversité des documents sélectionnés.

    L'objectif est d'éviter qu'une question globale
    soit représentée presque exclusivement par un seul PFE.
    """

    # ========================================================
    # 1. Embedding de la question
    # ========================================================

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    query_norm = np.linalg.norm(query_embedding)

    if query_norm > 0:
        query_embedding = query_embedding / query_norm


    # ========================================================
    # 2. Récupération des candidats avec FAISS
    # ========================================================

    candidate_k = min(candidate_k, len(chunks))

    distances, indices = index.search(
        query_embedding,
        candidate_k
    )

    candidate_indices = indices[0]
    candidate_distances = distances[0]


    # ========================================================
    # 3. Récupération des embeddings des candidats
    # ========================================================

    candidate_embeddings = np.array(
        [
            index.reconstruct(int(i))
            for i in candidate_indices
        ],
        dtype="float32"
    )


    # ========================================================
    # 4. Normalisation
    # ========================================================

    norms = np.linalg.norm(
        candidate_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    candidate_embeddings_normalized = (
        candidate_embeddings / norms
    )


    # ========================================================
    # 5. Similarité avec la question
    # ========================================================

    relevance_scores = (
        candidate_embeddings_normalized
        @ query_embedding[0]
    )


    # ========================================================
    # 6. Sélection MMR
    # ========================================================

    selected_positions = []

    remaining_positions = list(
        range(len(candidate_indices))
    )

    mmr_scores = {}

    # Documents déjà représentés
    selected_documents = set()

    # Poids de la couverture documentaire
    DOCUMENT_BONUS = 0.15


    while (
        remaining_positions
        and len(selected_positions) < top_k
    ):

        best_position = None
        best_score = -float("inf")


        for candidate_position in remaining_positions:

            # ------------------------------------------------
            # Pertinence
            # ------------------------------------------------

            relevance = relevance_scores[
                candidate_position
            ]


            # ------------------------------------------------
            # Redondance sémantique
            # ------------------------------------------------

            if not selected_positions:

                redundancy = 0.0

            else:

                candidate_vector = (
                    candidate_embeddings_normalized[
                        candidate_position
                    ]
                )

                selected_vectors = (
                    candidate_embeddings_normalized[
                        selected_positions
                    ]
                )

                similarities = (
                    selected_vectors
                    @ candidate_vector
                )

                redundancy = np.max(similarities)


            # ------------------------------------------------
            # Document du candidat
            # ------------------------------------------------

            index_position = candidate_indices[
                candidate_position
            ]

            candidate_document = chunks[
                index_position
            ]["metadata"]["source"]


            # ------------------------------------------------
            # Bonus si le document n'est pas encore représenté
            # ------------------------------------------------

            if candidate_document not in selected_documents:
                document_bonus = DOCUMENT_BONUS
            else:
                document_bonus = 0.0


            # ------------------------------------------------
            # Score final
            # ------------------------------------------------

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * redundancy
                + document_bonus
            )


            if mmr_score > best_score:

                best_score = mmr_score
                best_position = candidate_position


        # ----------------------------------------------------
        # Sélection du meilleur candidat
        # ----------------------------------------------------

        selected_positions.append(best_position)

        remaining_positions.remove(best_position)

        mmr_scores[best_position] = best_score


        # Ajouter le document aux documents représentés
        selected_index = candidate_indices[
            best_position
        ]

        selected_document = chunks[
            selected_index
        ]["metadata"]["source"]

        selected_documents.add(
            selected_document
        )


    # ========================================================
    # 7. Construction des résultats
    # ========================================================

    results = []

    for position in selected_positions:

        index_position = candidate_indices[
            position
        ]

        chunk = chunks[index_position]

        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "distance": float(
                candidate_distances[position]
            ),
            "mmr_score": float(
                mmr_scores[position]
            ),
            "index": int(index_position)
        })

    return results

# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

def display_results(results, title="RÉSULTATS"):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\n[{rank}]")
        print(
            f"Source : {metadata['source']}"
        )
        print(
            f"Page : {metadata['page']}"
        )
        print(
            f"Section : {metadata['section']}"
        )
        print(
            f"Distance FAISS : {result['distance']:.4f}"
        )

        if "mmr_score" in result:
            print(
                f"Score MMR : {result['mmr_score']:.4f}"
            )

        print(
            f"\n{result['text']}"
        )


# ============================================================
# TEST INTERACTIF
# ============================================================

def debug_candidates(
    query,
    model,
    index,
    chunks,
    candidate_k=15
):
    """
    Affiche les 15 premiers candidats FAISS
    avant l'application du MMR.
    """

    # Embedding de la question
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Recherche des candidats avec FAISS
    distances, indices = index.search(
        query_embedding,
        candidate_k
    )

    # Affichage
    print("\n" + "=" * 80)
    print(f"TOP {candidate_k} CANDIDATS FAISS")
    print("=" * 80)

    for rank, (distance, index_position) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):

        chunk = chunks[index_position]
        metadata = chunk["metadata"]

        print(f"\n[{rank}]")
        print(f"Distance : {distance:.4f}")
        print(f"Document : {metadata['source']}")
        print(
            f"Page : {metadata['page']} | "
            f"Section : {metadata['section']}"
        )

        print("\nTexte :")
        print(chunk["text"][:300])

        print("-" * 80)

def main():

    print("=" * 80)
    print("DIAGNOSTIC DU RETRIEVAL FAISS")
    print("=" * 80)

    print("\nChargement des ressources...")

    model, index, chunks = load_resources()

    print("Ressources chargées.")

    query = (
        "Quels sont les différents usages de "
        "l’intelligence artificielle représentés "
        "dans ces quatre sujets de PFE ?"
    )

    debug_candidates(
        query,
        model,
        index,
        chunks,
        candidate_k=15
    )


if __name__ == "__main__":
    main()

