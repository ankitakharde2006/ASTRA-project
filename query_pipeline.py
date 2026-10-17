"""
query_pipeline.py — The Naive RAG Online Query Pipeline.

The Goal:
    Accept a user's question live, convert it into an embedding vector via NVIDIA NIM,
    retrieve the top-k most relevant chunks from the Pinecone vector database,
    inject those chunks into an augmented prompt, and generate a grounded answer.

The "Why":
    This is the runtime loop that executes every time a user asks a question.
    By augmenting the model's prompt with retrieved private documents, we eliminate
    hallucinations and anchor the model's response directly to verified source facts.

Pipeline Stages Mapped to Day 4 Slide 13:
    5. embed_query()                 ---> Query Embedding (NVIDIA NIM)
    6. retrieve_top_k()               ---> Retrieval (Pinecone / Cosine Similarity)
    7. build_prompt_with_context()    ---> Context Injection (Prompt Augmentation)
    8. generate_answer()             ---> Generation (NVIDIA NIM LLM)
"""

import math
# pyrefly: ignore [missing-import]
from openai import OpenAI
from config import RAGConfig, get_rag_config
from vector_store import query_vector_db


def embed_query(
    query: str,
    config: RAGConfig,
) -> list[float]:
    """
    Stage 5: Query Embedding
    Embeds the user's live query into the same 2048-dimensional mathematical
    space used during the offline chunk indexing stage.
    Validates dimensional consistency (2048 dims) and handles API errors and rate limits.

    [TEACHING NOTE]
    Why must the query embedding model match the chunk embedding model?
    If you index your documents with Model A (e.g. 2048-dim) and embed your query with
    Model B (e.g. 1536-dim or different semantic coordinate system), the vector dimensions
    and coordinate meanings will not match! Always use the exact same embedding model.
    """
    from embedding import embed_query as nim_embed_query
    return nim_embed_query(query=query, config=config)



def retrieve_top_k(
    query_vector: list[float],
    config: RAGConfig,
) -> list[dict]:
    """
    Stage 6: Retrieval
    Queries the vector database for the top-k chunks with the highest
    cosine similarity to the query vector.
    """
    print("\n" + "=" * 60)
    print(f"STAGE 6: RETRIEVAL (TOP-{config.top_k} CHUNKS)")
    print("=" * 60)
    
    if config.use_local_vector_db:
        print("[*] Searching Database : Local Vector Store (JSON)")
    else:
        print(f"[*] Searching Database : Pinecone Index '{config.pinecone_index_name}'")

    matched_chunks = query_vector_db(
        query_vector=query_vector,
        top_k=config.top_k,
        config=config,
    )

    if len(matched_chunks) == 0:
        print("[!] No matching chunks found. Did you run indexer.py first?")
        return []

    print(f"[OK] Retrieved {len(matched_chunks)} most relevant passages:\n")

    rank = 1
    for match in matched_chunks:
        score_pct = round(match["score"] * 100.0, 2)
        print(f"  [Match #{rank}] ID: {match['id']} | Cosine Similarity: {match['score']:.4f} ({score_pct}%)")
        print(f"  Text Excerpt: \"{match['text']}\"\n")
        rank = rank + 1

    return matched_chunks


def build_prompt_with_context(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Stage 7: Context Injection (Prompt Augmentation)
    Assembles a structured system prompt combining the retrieved context passages
    with strict grounding instructions and the user's question.

    [TEACHING NOTE]
    The Grounding Instruction is the secret sauce of RAG. We explicitly instruct the
    LLM to ONLY use the provided context and never speculate. This anchors the model's
    probabilistic decoder to real facts.
    """
    print("\n" + "=" * 60)
    print("STAGE 7: CONTEXT INJECTION (AUGMENTED PROMPT)")
    print("=" * 60)

    # Format the retrieved context blocks explicitly
    context_blocks = []
    chunk_num = 1
    for chunk in retrieved_chunks:
        header = f"--- CONTEXT DOCUMENT {chunk_num} (ID: {chunk['id']}, Relevance: {chunk['score']:.4f}) ---"
        body = chunk["text"]
        context_blocks.append(f"{header}\n{body}")
        chunk_num = chunk_num + 1

    combined_context = "\n\n".join(context_blocks)

    augmented_prompt = f"""You are the official HR & Workplace AI assistant for AstraTech Solutions.
Answer the user's inquiry accurately, professionally, and EXCLUSIVELY based on the verified reference context below.
If the answer cannot be determined strictly from the context, respond: "I do not have sufficient information in the AstraTech employee handbook to answer that question."
Do not speculate, assume, or hallucinate outside the given text.

=== BEGIN VERIFIED REFERENCE CONTEXT ===
{combined_context}
=== END VERIFIED REFERENCE CONTEXT ===

USER QUESTION:
{query}

GROUNDED ANSWER:"""

    print(f"[*] Combined Context Length : {len(combined_context)} characters across {len(retrieved_chunks)} passages")
    print(f"[*] Total Augmented Prompt   : {len(augmented_prompt)} characters")
    print("[OK] Prompt assembled with grounded context injection.")
    
    return augmented_prompt


def generate_answer(
    prompt: str,
    config: RAGConfig,
) -> str:
    """
    Stage 8: Generation (Grounded LLM Call via NVIDIA NIM)
    Sends the context-augmented prompt to the LLM to synthesize the final answer.
    """
    print("\n" + "=" * 60)
    print("STAGE 8: GENERATION (NVIDIA NIM)")
    print("=" * 60)
    print(f"[*] Generation Model : {config.chat_model}")
    print("[*] Generating grounded response...")

    # Branch 1: Live NVIDIA NIM Chat Completion API
    if not config.use_mock_ai:
        client = OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a precise technical AI assistant grounded strictly in provided operational manuals.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = client.chat.completions.create(
            model=config.chat_model,
            messages=messages,
            temperature=0.1,  # Low temperature for deterministic, factual retrieval
            max_tokens=300,
        )

        raw_answer = response.choices[0].message.content
        if raw_answer is None:
            # Check reasoning content if present
            raw_answer = getattr(response.choices[0].message, "reasoning_content", "") or ""

        final_answer = raw_answer.strip()
        print("[OK] Response received from NVIDIA NIM:")
        print("-" * 60)
        print(final_answer)
        print("-" * 60)
        return final_answer

    # Branch 2: Simulated Offline Response
    print("[!] Offline mode active. Returning simulated grounded answer.")
    mock_reply = (
        "[SIMULATED RAG ANSWER] The one-time remote home office setup stipend at "
        "AstraTech Solutions is $1,200 (to be submitted via Expensify within 30 days). "
        "The office guest Wi-Fi network SSID is AstraTech-Guest with password 'CoffeeAndCode2026!'."
    )
    print("-" * 60)
    print(mock_reply)
    print("-" * 60)
    return mock_reply


def generate_answer_without_rag(
    query: str,
    config: RAGConfig,
) -> str:
    """
    Teaching Comparison Function:
    Ask the LLM the exact same question WITHOUT providing any retrieved context.
    
    [TEACHING NOTE]
    Running this in class shows students why RAG is necessary. Because AstraTech's
    employee handbook is private internal documentation, the base LLM has never seen it in its training data.
    Without RAG, the model either refuses or hallucinates a completely made-up answer!
    """
    print("\n" + "#" * 60)
    print("BASELINE COMPARISON: RUNNING QUERY WITHOUT RAG (RAW LLM)")
    print("#" * 60)
    print(f"[*] Query Sent to LLM : \"{query}\"")
    print("[*] (Notice: No documents or retrieved context are provided in this call!)")

    if not config.use_mock_ai:
        client = OpenAI(
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer the user's question directly.",
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        try:
            response = client.chat.completions.create(
                model=config.chat_model,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
            )
            raw_answer = response.choices[0].message.content
            if raw_answer is None:
                raw_answer = getattr(response.choices[0].message, "reasoning_content", "") or ""
            
            answer = raw_answer.strip()
            print("\n[RAW LLM RESPONSE (WITHOUT RAG)]:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            return answer
        except Exception as err:
            err_msg = f"[!] Raw LLM call failed: {err}"
            print(err_msg)
            return err_msg

    mock_uninformed = (
        "[RAW LLM ANSWER WITHOUT RAG]: I do not have access to internal company policies or "
        "employee handbooks for AstraTech Solutions."
    )
    print("\n[RAW LLM RESPONSE (WITHOUT RAG)]:")
    print("-" * 60)
    print(mock_uninformed)
    print("-" * 60)
    return mock_uninformed


def run_query_pipeline(
    query: str,
    config: RAGConfig | None = None,
) -> dict:
    """
    Execute the complete 4-stage Query Pipeline sequentially:
    5. Query Embedding
    6. Retrieval
    7. Context Injection
    8. Generation
    """
    if config is None:
        config = get_rag_config()

    print("\n" + "#" * 60)
    print("STARTING NAIVE RAG QUERY PIPELINE")
    print("#" * 60)

    # 5. Query Embedding
    query_vector = embed_query(query=query, config=config)

    # 6. Retrieval
    retrieved_chunks = retrieve_top_k(query_vector=query_vector, config=config)

    # 7. Context Injection
    augmented_prompt = build_prompt_with_context(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    # 8. Generation
    answer = generate_answer(
        prompt=augmented_prompt,
        config=config,
    )

    print("\n" + "#" * 60)
    print("[SUCCESS] QUERY PIPELINE FINISHED -- ANSWER PRODUCED!")
    print("#" * 60 + "\n")

    result = {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "answer": answer,
    }
    return result


if __name__ == "__main__":
    # Test query pipeline with the canonical live test question
    cfg = get_rag_config()
    test_question = (
        "How much is the one-time remote home office setup stipend, "
        "and what is the Wi-Fi password for office visitors?"
    )
    
    # 1. Run baseline without RAG
    generate_answer_without_rag(query=test_question, config=cfg)
    
    # 2. Run with RAG
    run_query_pipeline(query=test_question, config=cfg)
