import argparse
import os
import shutil
from langchain.schema.document import Document
from embedding import get_embedding_function
from langchain_chroma import Chroma  
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from langchain_community.vectorstores.utils import filter_complex_metadata
from ftfy import fix_text
CHROMA_PATH = "chroma"
DATA_PATH = "data"
EMBED_MODEL_ID = "cnmoro/Qwen3-0.6B-Portuguese-Tokenizer"
MAX_TOKENS = 32768  # set to a small number for illustrative purposes


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="limpar a base.")
    args = parser.parse_args()
    if args.reset:
        print("limpar a base")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    source ="C:/Users/mauri/OneDrive/Área de Trabalho/AI Agent/data/estudotecnicasdesborramentoimagem.pdf"  # PDF path or URL
    converter = DocumentConverter()
    result = converter.convert(source)
    for text_obj in result.document.texts:
        if hasattr(text_obj, "text") and isinstance(text_obj.text, str):
            text_obj.text = fix_text(text_obj.text)

    return result.document

def split_documents(documents: list[Document]):
    tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS)
    chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,  # optional, defaults to True
)
    chunk_iter = chunker.chunk(dl_doc=documents)
    chunks = list(chunk_iter)
    return chunks


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # Só IDs
    existing_ids = set(existing_items["ids"])
    print(f"Documentos no DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id and chunk_id not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"Adicionando novos documentos: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("Sem documentos")

def clean_metadata(metadata: dict) -> dict:
    if not metadata:
        return {}

    clean_meta = {}

    origin = metadata.get("origin", {})
    clean_meta["filename"] = origin.get("filename") if isinstance(origin, dict) else None

    page_numbers = []
    for item in metadata.get("doc_items", []):
        for prov in item.get("prov", []):
            page_no = prov.get("page_no")
            if isinstance(page_no, int):
                page_numbers.append(page_no)

    # Converte a lista em string
    clean_meta["page_numbers"] = (
        ",".join(str(p) for p in sorted(set(page_numbers)))
        if page_numbers else None
    )

    chunk_id = metadata.get("chunk_id")
    if isinstance(chunk_id, str):
        clean_meta["chunk_id"] = chunk_id

    return clean_meta


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    chunks_with_ids = []

    for chunk in chunks:
        source = chunk.meta.origin.filename
        page = chunk.meta.doc_items[0].prov[0].page_no
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk_dict = chunk.model_dump()
        chunk_text = chunk.text if hasattr(chunk, "text") else chunk_dict.get("text", "")
        metadata = chunk_dict.get("meta", {})
        metadata["chunk_id"] = chunk_id
        clean_meta  = clean_metadata(metadata)
        # Certifique-se que doc é um Document, não um tuple
        doc = Document(page_content=chunk_text, metadata=clean_meta)
        print(type(doc))
        print(doc)
        chunks_with_ids.append(doc)

    return chunks_with_ids

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()