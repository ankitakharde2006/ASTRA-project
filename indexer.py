"""
indexer.py — The Naive RAG Offline Indexing Pipeline.

The Goal:
    Ingest a source knowledge document, split it into overlapping chunks,
    convert each chunk into a high-dimensional vector via NVIDIA NIM, and
    persist the vectors and metadata into the Pinecone Vector Database.

The "Why":
    Language models have a fixed knowledge cutoff and cannot memorize proprietary
    or private company manuals without fine-tuning (which is slow and expensive).
    The indexing pipeline prepares external knowledge into indexed vector coordinates
    so relevant information can be fetched in milliseconds during live queries.

Pipeline Stages Mapped to Day 4 Slide 13:
    1. load_document()       ---> Document Ingestion
    2. chunk_document()      ---> Chunking
    3. embed_chunks()        ---> Embedding (NVIDIA NIM)
    4. store_in_vector_db()  ---> Storage (Pinecone Serverless)
"""

import math
import os
from pathlib import Path
from openai import OpenAI
from config import RAGConfig, get_rag_config
from vector_store import upsert_to_vector_db


EXCLUDED_DIRECTORIES = {
    ".venv",
    "venv",
    "env",
    ".env",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "node_modules",
    ".idea",
    ".vscode",
    "htmlcov",
    ".coverage",
}


def is_excluded_directory(dir_name: str) -> bool:
    """
    Check if a directory name should be excluded from Python file indexing.
    Excludes virtual environments, version control, caches, and generated folders.
    """
    normalized = dir_name.lower().strip()
    if normalized in EXCLUDED_DIRECTORIES:
        return True
    if normalized.startswith("."):
        return True
    if normalized.startswith("__") and normalized.endswith("__"):
        return True
    if normalized.endswith(".egg-info"):
        return True
    return False


def load_python_files(
    directory: str | Path | None = None,
    exclude_dirs: set[str] | list[str] | None = None,
    include_empty: bool = False,
) -> list[dict]:
    """
    Stage 1: Python Source Files Ingestion
    Recursively scans the target project directory for relevant .py files,
    excluding .venv, .git, __pycache__, and other generated or temporary folders.
    Preserves each file's relative path alongside its full content.
    Handles empty and unreadable files safely without crashing.
    Does not read or print .env values.

    Returns:
        list[dict]: A list of records, where each record contains:
            - 'file_path': relative path string
            - 'relative_path': relative path string
            - 'path': relative path string
            - 'content': full file content string
            - 'text': full file content string
            - 'line_count': number of lines
            - 'char_count': number of characters
    """
    if directory is None:
        target_path = Path(__file__).resolve().parent
    else:
        target_path = Path(directory).resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"Target directory or file not found at: {target_path}")

    # Handle edge case where a single .py file is provided
    if target_path.is_file():
        if not target_path.name.endswith(".py"):
            return []
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                single_content = f.read()
        except Exception as err:
            print(f"[!] Warning: Could not read single file {target_path.name}: {err}")
            return []

        if len(single_content.strip()) == 0 and not include_empty:
            print(f"[*] Skipping empty file: {target_path.name}")
            return []

        return [{
            "file_path": target_path.name,
            "relative_path": target_path.name,
            "path": target_path.name,
            "content": single_content,
            "text": single_content,
            "line_count": len(single_content.splitlines()),
            "char_count": len(single_content),
        }]

    custom_excludes = set(exclude_dirs) if exclude_dirs else set()

    def _is_dir_excluded(d_name: str) -> bool:
        if d_name in custom_excludes:
            return True
        return is_excluded_directory(d_name)

    loaded_files: list[dict] = []
    total_lines = 0
    total_chars = 0

    print("\n" + "=" * 60)
    print("STAGE 1: PYTHON SOURCE CODE INGESTION")
    print("=" * 60)
    print(f"[*] Target Directory : {target_path}")

    for root, dirs, files in os.walk(target_path):
        # Prune excluded directories in-place so os.walk avoids recursing into them
        dirs[:] = [d for d in sorted(dirs) if not _is_dir_excluded(d)]

        for file_name in sorted(files):
            # Only read relevant .py files
            if not file_name.endswith(".py"):
                continue

            # Ignore hidden files
            if file_name.startswith("."):
                continue

            full_file_path = Path(root) / file_name

            try:
                rel_path = full_file_path.relative_to(target_path)
            except ValueError:
                rel_path = full_file_path

            rel_path_str = str(rel_path)

            # Defensive verification: check all parent path segments
            if any(_is_dir_excluded(part) for part in rel_path.parts[:-1]):
                continue

            # Safe handling of empty and unreadable files
            try:
                # Check file size metadata
                st_size = full_file_path.stat().st_size
                if st_size == 0 and not include_empty:
                    print(f"[*] Handled empty file safely (0 bytes, skipped): {rel_path_str}")
                    continue

                with open(full_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            except UnicodeDecodeError:
                # Try fallback decoding before gracefully skipping
                try:
                    with open(full_file_path, "r", encoding="utf-8-sig") as f:
                        content = f.read()
                except Exception as decode_err:
                    print(f"[!] Warning: Handled unreadable file safely (decode error): {rel_path_str} ({decode_err})")
                    continue
            except (PermissionError, OSError, IOError) as io_err:
                print(f"[!] Warning: Handled unreadable file safely (I/O error): {rel_path_str} ({io_err})")
                continue

            # Check whitespace-only content
            if len(content.strip()) == 0 and not include_empty:
                print(f"[*] Handled empty file safely (whitespace-only, skipped): {rel_path_str}")
                continue

            lines = content.splitlines()
            line_cnt = len(lines)
            char_cnt = len(content)
            total_lines += line_cnt
            total_chars += char_cnt

            doc_record = {
                "file_path": rel_path_str,
                "relative_path": rel_path_str,
                "path": rel_path_str,
                "content": content,
                "text": content,
                "line_count": line_cnt,
                "char_count": char_cnt,
            }
            loaded_files.append(doc_record)
            print(f"  -> [{rel_path_str}] ({line_cnt} lines, {char_cnt} chars)")

    print(f"[OK] Ingestion stage complete: {len(loaded_files)} Python files loaded ({total_lines} lines, {total_chars} chars).")
    return loaded_files


# Convenient aliases for function callers
read_python_files = load_python_files
load_source_files = load_python_files


def load_document(file_path: str) -> str:
    """
    Stage 1: Document Ingestion (Single Document)
    Read the raw text content from the specified source document file.
    Preserved for backwards compatibility with existing project demo runs.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Document file not found at: {file_path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_text = text.strip()
    
    print("\n" + "=" * 60)
    print("STAGE 1: DOCUMENT INGESTION")
    print("=" * 60)
    print(f"[*] Source File Path : {path_obj.resolve()}")
    print(f"[*] Total Characters : {len(cleaned_text)}")
    print(f"[*] Total Words      : {len(cleaned_text.split())}")
    print("[OK] Document loaded into memory successfully.")
    
    return cleaned_text


def chunk_document(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 50,
    file_path: str = "",
) -> list[dict]:
    """
    Stage 2: Chunking (Fixed-size sliding window)
    Splits the continuous text string into overlapping discrete passages.

    Preserves source file_path and chunk_index on each chunk record.
    Avoids empty chunks and prevents infinite loops with strict validation.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer greater than 0, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative (>= 0), got {chunk_overlap}.")
    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be strictly greater than chunk_overlap ({chunk_overlap}) "
            f"to prevent an infinite loop."
        )

    step = chunk_size - chunk_overlap
    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    print("\n" + "=" * 60)
    print("STAGE 2: CHUNKING (SLIDING WINDOW)")
    print("=" * 60)
    print(f"[*] Window Size    : {chunk_size} characters")
    print(f"[*] Window Overlap : {chunk_overlap} characters")
    print(f"[*] Stride / Step  : {step} characters")
    if file_path:
        print(f"[*] Source File    : {file_path}")

    while start < len(text):
        end = start + chunk_size
        chunk_slice = text[start:end]
        
        # Clean leading and trailing whitespace
        trimmed_chunk = chunk_slice.strip()
        
        if len(trimmed_chunk) > 0:
            chunk_id = f"{file_path}#chunk_{chunk_index}" if file_path else f"doc_chunk_{chunk_index}"
            chunk_record = {
                "id": chunk_id,
                "file_path": file_path,
                "relative_path": file_path,
                "source_file": file_path,
                "chunk_index": chunk_index,
                "text": trimmed_chunk,
                "content": trimmed_chunk,
                "start_char": start,
                "end_char": start + len(chunk_slice),
                "char_count": len(trimmed_chunk),
            }
            chunks.append(chunk_record)
            
            # Print preview for live lecture visibility
            preview = trimmed_chunk.replace("\n", " ")
            if len(preview) > 65:
                preview = preview[:65] + "..."
            print(f"  -> [{chunk_record['id']}] ({len(trimmed_chunk)} chars): \"{preview}\"")
            chunk_index = chunk_index + 1

        start = start + step

    print(f"[OK] Created {len(chunks)} discrete chunks from source document.")
    return chunks


def chunk_files(
    files: list[dict] | str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Stage 2: Chunking for Ingested Source Code Files
    Takes a list of file records (from load_python_files) or a directory path,
    and breaks each source file into overlapping chunks with configurable
    chunk_size and chunk_overlap settings.

    Preserves source file_path and chunk_index for every chunk.
    Avoids empty chunks and prevents infinite loops.
    Does not use LangChain or LlamaIndex.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer greater than 0, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative (>= 0), got {chunk_overlap}.")
    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be strictly greater than chunk_overlap ({chunk_overlap}) "
            f"to prevent an infinite loop."
        )

    # If a path or directory string is passed, invoke the Stage 1 reader
    if isinstance(files, (str, Path)):
        files = load_python_files(files)

    all_chunks: list[dict] = []
    
    print("\n" + "=" * 60)
    print("STAGE 2: CHUNKING SOURCE FILES (SLIDING WINDOW)")
    print("=" * 60)
    print(f"[*] Window Size    : {chunk_size} characters")
    print(f"[*] Window Overlap : {chunk_overlap} characters")
    print(f"[*] Stride / Step  : {chunk_size - chunk_overlap} characters")
    print(f"[*] Files to Chunk : {len(files)}")

    for file_record in files:
        file_path = (
            file_record.get("file_path")
            or file_record.get("relative_path")
            or file_record.get("path")
            or "unknown_source.py"
        )
        content = file_record.get("content") or file_record.get("text") or ""

        # Safely skip empty file contents
        if not content.strip():
            print(f"[*] Skipping empty file content for: {file_path}")
            continue

        file_chunks = chunk_document(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path=file_path,
        )
        all_chunks.extend(file_chunks)

    print(f"[OK] Ingestion chunking complete: {len(all_chunks)} discrete chunks generated across {len(files)} files.")
    return all_chunks


# Convenient aliases
chunk_python_files = chunk_files
chunk_documents = chunk_files


def embed_chunks(
    chunks: list[dict],
    config: RAGConfig,
) -> list[dict]:
    """
    Stage 3: Embedding (NVIDIA NIM)
    Converts each textual chunk into a dense 2048-dimensional vector representation.
    Validates dimensional consistency (2048 dims), handles rate limits with retries,
    and guards against malformed responses.

    [TEACHING NOTE]
    Embedding models translate semantic concepts into mathematical space.
    Chunks with similar meaning will be close together in this 2048-dimensional space,
    even if they use different words or synonyms!
    """
    from embedding import embed_chunks as nim_embed_chunks
    return nim_embed_chunks(chunks=chunks, config=config)



def store_in_vector_db(
    chunks_with_embeddings: list[dict],
    config: RAGConfig,
) -> int:
    """
    Stage 4: Storage
    Persist chunk vectors and text metadata into the Pinecone Vector Database.
    """
    print("\n" + "=" * 60)
    print("STAGE 4: STORAGE IN VECTOR DATABASE")
    print("=" * 60)
    if config.use_local_vector_db:
        print("[*] Target Storage : Local JSON Vector Store (Pinecone Fallback)")
    else:
        print(f"[*] Target Storage : Pinecone Serverless Index '{config.pinecone_index_name}'")
        print(f"[*] Cloud Provider : {config.pinecone_cloud.upper()} ({config.pinecone_region})")

    stored_count = upsert_to_vector_db(
        chunks_with_embeddings=chunks_with_embeddings,
        config=config,
    )

    print(f"[OK] Storage stage complete: {stored_count} records indexed and ready for retrieval.")
    return stored_count


def run_indexing_pipeline(
    file_path: str,
    config: RAGConfig | None = None,
) -> list[dict]:
    """
    Execute the complete 4-stage Indexing Pipeline sequentially:
    1. Document Ingestion
    2. Chunking
    3. Embedding
    4. Storage
    """
    if config is None:
        config = get_rag_config()

    print("\n" + "#" * 60)
    print("STARTING NAIVE RAG INDEXING PIPELINE")
    print("#" * 60)

    # 1. Ingestion
    raw_document = load_document(file_path=file_path)

    # 2. Chunking
    chunk_list = chunk_document(
        text=raw_document,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # 3. Embedding
    embedded_chunks = embed_chunks(
        chunks=chunk_list,
        config=config,
    )

    # 4. Storage
    store_in_vector_db(
        chunks_with_embeddings=embedded_chunks,
        config=config,
    )

    print("\n" + "#" * 60)
    print("[SUCCESS] INDEXING PIPELINE FINISHED -- KNOWLEDGE BASE READY!")
    print("#" * 60 + "\n")

    return embedded_chunks


if __name__ == "__main__":
    import sys
    # If invoked with --stage1, execute only the Stage 1 Python source file reader
    if "--stage1" in sys.argv:
        target_dir_arg = None
        for arg in sys.argv[1:]:
            if arg != "--stage1" and not arg.startswith("-"):
                target_dir_arg = arg
                break
        ingested = load_python_files(directory=target_dir_arg)
        print(f"\n[STAGE 1 VERIFICATION] Ingested {len(ingested)} Python files successfully.")
    elif "--stage2" in sys.argv or "--chunk" in sys.argv:
        target_dir_arg = None
        for arg in sys.argv[1:]:
            if arg not in ("--stage2", "--chunk") and not arg.startswith("-"):
                target_dir_arg = arg
                break
        ingested = load_python_files(directory=target_dir_arg)
        chunks = chunk_files(ingested)
        print(f"\n[STAGE 2 VERIFICATION] Successfully generated {len(chunks)} chunks across {len(ingested)} Python files.")
    elif "--stage3" in sys.argv or "--embed" in sys.argv:
        target_dir_arg = None
        for arg in sys.argv[1:]:
            if arg not in ("--stage3", "--embed") and not arg.startswith("-"):
                target_dir_arg = arg
                break
        ingested = load_python_files(directory=target_dir_arg)
        chunks = chunk_files(ingested)
        cfg = get_rag_config()
        # Test embedding on the first 3 chunks to confirm NIM live embedding and dimensions
        test_sample = chunks[:3] if len(chunks) >= 3 else chunks
        embedded = embed_chunks(test_sample, config=cfg)
        print(f"\n[STAGE 3 VERIFICATION] Successfully embedded {len(embedded)} chunks via NVIDIA NIM.")
        if embedded:
            dim = len(embedded[0]["embedding"])
            print(f"[*] Confirmed vector dimension: {dim} (Expected: 2048)")
    else:
        # Default: preserve original standalone demo
        cfg = get_rag_config()
        sample_doc = str(Path(__file__).parent / "data" / "astratech_employee_handbook.txt")
        run_indexing_pipeline(file_path=sample_doc, config=cfg)

