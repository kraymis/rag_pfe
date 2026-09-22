import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import load_resources, search_mmr

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError(
        "NVIDIA_API_KEY n'est pas définie dans le fichier .env"
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=90.0,
    max_retries=1,
)

MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
TOP_K = 5


# ============================================================
# CONSTRUCTION DU CONTEXTE
# ============================================================

def build_context(results):
    context_parts = []

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        context_parts.append(
            f"""
[Source {rank}]

Entreprise : {metadata["entreprise"]}
Année : {metadata["annee"]}
Référence : {metadata["reference"]}
Domaine : {metadata["domaine"]}
Durée : {metadata["duree"]}
Statut : {metadata["statut"]}

Page : {metadata["page"]}
Section : {metadata["section"]}

Passage :
{result["text"]}
"""
        )

    return "\n".join(context_parts)

# ============================================================
# GÉNÉRATION DE LA RÉPONSE
# ============================================================

def generate_answer(question, results):

    context = build_context(results)

    system_prompt = """
Tu es un assistant chargé de répondre à des questions
à partir d'un corpus de sujets de PFE.

RÈGLES IMPORTANTES :

1. Réponds uniquement à partir des informations présentes
   dans le contexte fourni, y compris les métadonnées.

2. N'invente aucune information.

3. Si l'information nécessaire pour répondre à la question
   n'est réellement pas présente dans le contexte, réponds
   UNE SEULE FOIS :

   "Cette information n'est pas disponible dans les documents fournis."

4. Si l'information est présente, ne dis PAS qu'elle est
   indisponible.

5. Réponds directement à la question et ne répète pas
   inutilement la même information.

6. Lorsque tu réponds, cite les sources utilisées sous la forme :
   [Source 1], [Source 2], etc.

7. Si plusieurs documents sont nécessaires pour répondre,
   utilise-les tous.

8. Pour une question portant sur l'apprentissage à partir
   de données, considère comme apprentissage automatique
   les méthodes telles que la classification, la régression,
   la détection d'anomalies, les arbres de décision,
   le Random Forest, le Gradient Boosting, les CNN,
   le Deep Learning ou le transfer learning.

   Le simple calcul d'embeddings, l'indexation vectorielle
   ou la recherche par similarité ne constituent pas,
   à eux seuls, un apprentissage réalisé à partir des données
   du projet.

9. Réponds en français de manière claire, concise et factuelle.

10. Ne produis qu'une seule réponse finale.
"""

    user_prompt = f"""
QUESTION :
{question}

CONTEXTE :
{context}

RÉPONSE :
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        top_p=1,
        max_tokens=1024,
        stream=False
    )

    return completion.choices[0].message.content


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 80)
    print("RAG - GÉNÉRATION DE RÉPONSE")
    print("=" * 80)

    print("\nChargement des ressources...")

    model, index, chunks = load_resources()

    print("Ressources chargées.")

    while True:

        question = input(
            "\nVotre question (ou 'quit' pour quitter) : "
        )

        if question.lower() == "quit":
            break

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        results = search_mmr(
            question,
            model,
            index,
            chunks,
            top_k=TOP_K,
            candidate_k=15,
            lambda_param=0.7
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        print("\nGénération de la réponse...")

        try:

            answer = generate_answer(
                question,
                results
            )

        except Exception as error:

            print("\nErreur lors de l'appel à NVIDIA :")
            print(error)
            continue

        # ----------------------------------------------------
        # AFFICHAGE
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("RÉPONSE")
        print("=" * 80)

        print(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("SOURCES RÉCUPÉRÉES")
        print("-" * 80)

        for rank, result in enumerate(results, start=1):

            metadata = result["metadata"]

            print(
                f"[Source {rank}] "
                f"{metadata['source']} | "
                f"Page {metadata['page']} | "
                f"Section : {metadata['section']} | "
                f"Distance : {result['distance']:.4f}"
            )


if __name__ == "__main__":
    main()