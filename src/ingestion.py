import pymupdf
from pathlib import Path
import re

DATA_DIR = Path("data")

METADATA_FIELDS = {
    "entreprise": ["entreprise"],
    "annee": ["année", "annee"],
    "reference": ["référence", "reference"],
    "domaine": ["domaine"],
    "duree": ["durée indicative", "duree indicative", "durée", "duree"],
    "statut": ["statut"],
}

CONTENT_SECTIONS = {
    "contexte",
    "objectif du projet",
    "informations et connaissances disponibles",
    "données disponibles",
    "jeu de données",
    "corpus documentaire",
    "travaux à réaliser",
    "approche méthodologique",
    "résultats attendus",
    "mots-clés",
}


def clean_text(text):
    """Nettoie légèrement le texte sans supprimer son contenu."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)

        lines.append(line)

    return "\n".join(lines)


def normalize(text):
    """Normalise un texte pour faciliter les comparaisons."""
    text = text.strip().lower()

    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "ä": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def extract_metadata(text):
    """
    Extrait les métadonnées présentes dans l'en-tête du document.

    La fonction se base sur les noms génériques des champs
    et sur la structure du document, sans dépendre d'une phrase
    particulière du contenu.
    """

    metadata = {
        "entreprise": None,
        "annee": None,
        "reference": None,
        "domaine": None,
        "duree": None,
        "statut": None,
    }

    cleaned_text = clean_text(text)

    lines = [
        line.strip()
        for line in cleaned_text.split("\n")
        if line.strip()
    ]

    field_mapping = {}

    for metadata_name, possible_names in METADATA_FIELDS.items():
        for name in possible_names:
            field_mapping[normalize(name)] = metadata_name

    normalized_sections = {
        normalize(section)
        for section in CONTENT_SECTIONS
    }

    current_field = None
    current_value = []
    header_started = False

    def save_field():
        nonlocal current_field
        nonlocal current_value

        if current_field is None:
            return

        value = " ".join(current_value).strip()

        if not value:
            return

        if current_field == "annee":
            match = re.search(r"\b(19|20)\d{2}\b", value)

            if match:
                metadata[current_field] = int(match.group())

        else:
            metadata[current_field] = value

    for line in lines:

        normalized_line = normalize(line)

        # Une section de contenu signifie que l'en-tête est terminé.
        if (
            header_started
            and normalized_line in normalized_sections
        ):
            save_field()
            break

        # Nouveau champ de métadonnée.
        if normalized_line in field_mapping:
            save_field()

            current_field = field_mapping[normalized_line]
            current_value = []
            header_started = True

        # Suite de la valeur du champ courant.
        elif current_field is not None:
            current_value.append(line)

    save_field()

    return metadata


def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte du PDF page par page et associe
    les mêmes métadonnées documentaires à chaque page.
    """

    document = pymupdf.open(pdf_path)

    all_text = []

    for page in document:
        all_text.append(page.get_text())

    full_text = "\n".join(all_text)

    document_metadata = extract_metadata(full_text)

    pages = []

    for page_number, page in enumerate(document):

        text = clean_text(page.get_text())

        if not text:
            continue

        pages.append({
            "text": text,
            "metadata": {
                "source": pdf_path.stem,
                "page": page_number + 1,
                **document_metadata
            }
        })

    document.close()

    return pages


def main():

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Aucun fichier PDF trouvé dans le dossier data/")
        return

    for pdf_path in pdf_files:

        print("\n" + "=" * 80)
        print(f"DOCUMENT : {pdf_path.name}")
        print("=" * 80)

        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print("Aucune page exploitable.")
            continue

        metadata = pages[0]["metadata"]

        print("\nMÉTADONNÉES")
        print("-" * 80)

        for key, value in metadata.items():
            print(f"{key:<12}: {value}")

        print(f"\nNombre de pages : {len(pages)}")


if __name__ == "__main__":
    main()