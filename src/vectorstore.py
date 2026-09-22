import faiss
import numpy as np
from pathlib import Path


INDEX_DIR = Path("index")

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"


def main():

    # 1. Charger les embeddings
    print("Chargement des embeddings...")

    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Nombre de vecteurs : {embeddings.shape[0]}")
    print(f"Dimension : {embeddings.shape[1]}")

    # 2. S'assurer que les embeddings sont en float32
    embeddings = embeddings.astype("float32")

    # 3. Créer l'index FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # 4. Ajouter les embeddings à l'index
    index.add(embeddings)

    print(f"Nombre de vecteurs dans FAISS : {index.ntotal}")

    # 5. Sauvegarder l'index
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Index FAISS sauvegardé dans : {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()