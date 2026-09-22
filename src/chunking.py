from ingestion import extract_text_from_pdf, CONTENT_SECTIONS, normalize
from pathlib import Path
import re

DATA_DIR = Path("data")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_into_sentences(text):
    """Découpe un texte en phrases."""
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Découpe un texte en chunks en conservant des phrases complètes.
    """

    sentences = split_into_sentences(text)

    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        if (
            current_chunk
            and current_length + sentence_length > chunk_size
        ):

            chunk_text = " ".join(current_chunk).strip()

            chunks.append(chunk_text)

            # Construction du chevauchement.
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_chunk):

                if overlap_length + len(previous_sentence) <= overlap:

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_length += len(previous_sentence)

                else:
                    break

            current_chunk = overlap_sentences + [sentence]

            current_length = sum(
                len(sentence)
                for sentence in current_chunk
            ) + max(len(current_chunk) - 1, 0)

        else:

            current_chunk.append(sentence)

            current_length += (
                sentence_length
                + (1 if current_chunk else 0)
            )

    if current_chunk:
        chunks.append(
            " ".join(current_chunk).strip()
        )

    return chunks


def identify_section(line):
    """
    Retourne le nom de la section si la ligne correspond
    à une section connue.
    """

    normalized_line = normalize(line)

    for section in CONTENT_SECTIONS:

        if normalized_line == normalize(section):
            return section

    return None


def remove_document_header(lines):
    """
    Supprime uniquement les éléments administratifs du début
    du document avant la première section de contenu.

    Les métadonnées sont déjà conservées séparément.
    """

    content_start = None

    for i, line in enumerate(lines):

        if identify_section(line) is not None:
            content_start = i
            break

    if content_start is None:
        return lines

    return lines[content_start:]


def extract_sections_from_page(page_text):
    """
    Transforme le texte d'une page en sections.

    Retourne une liste de :
    {
        "section": "...",
        "text": "..."
    }
    """

    lines = [
        line.strip()
        for line in page_text.split("\n")
        if line.strip()
    ]

    if not lines:
        return []

    sections = []

    current_section = None
    current_lines = []

    for line in lines:

        section_name = identify_section(line)

        if section_name is not None:

            # Sauvegarder la section précédente.
            if current_section is not None and current_lines:

                sections.append({
                    "section": current_section,
                    "text": " ".join(current_lines).strip()
                })

            current_section = section_name
            current_lines = []

        elif current_section is not None:

            current_lines.append(line)

    # Sauvegarder la dernière section.
    if current_section is not None and current_lines:

        sections.append({
            "section": current_section,
            "text": " ".join(current_lines).strip()
        })

    return sections


def create_chunks_from_pages(pages):
    """
    Crée les chunks à partir des pages extraites.

    Le chunking est effectué à l'intérieur de chaque section.
    """

    all_chunks = []

    for page in pages:

        page_text = page["text"]

        sections = extract_sections_from_page(page_text)

        for section in sections:

            section_name = section["section"]
            section_text = section["text"]

            chunks = create_chunks(section_text)

            for chunk_number, chunk_text in enumerate(
                chunks,
                start=1
            ):

                all_chunks.append({
                    "text": chunk_text,

                    "metadata": {
                        **page["metadata"],
                        "section": section_name,
                        "chunk": chunk_number
                    }
                })

    return all_chunks


def main():

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Aucun fichier PDF trouvé dans le dossier data/")
        return

    all_chunks = []

    for pdf_path in pdf_files:

        print("\n" + "=" * 80)
        print(f"DOCUMENT : {pdf_path.name}")
        print("=" * 80)

        pages = extract_text_from_pdf(pdf_path)

        document_chunks = create_chunks_from_pages(pages)

        all_chunks.extend(document_chunks)

        print(
            f"Nombre de chunks pour ce document : "
            f"{len(document_chunks)}"
        )

        for i, chunk in enumerate(
            document_chunks,
            start=1
        ):

            metadata = chunk["metadata"]

            print("\n" + "-" * 80)
            print(f"CHUNK {i}")
            print(f"Section : {metadata['section']}")
            print(f"Page    : {metadata['page']}")
            print(f"Taille  : {len(chunk['text'])} caractères")
            print("-" * 80)
            print(chunk["text"])

    print("\n" + "=" * 80)
    print(f"TOTAL : {len(all_chunks)} chunks")
    print("=" * 80)


if __name__ == "__main__":
    main()