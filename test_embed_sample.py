"""
test_embed_sample.py — Small verification test for NVIDIA NIM embedding stage.

Embeds one harmless sample string, validates that the returned vector has
exactly 2048 dimensions, and reports the length without printing secrets.
"""

from embedding import embed_text, EXPECTED_DIMENSION
from config import get_rag_config


def main():
    # 1. Load existing configuration safely without printing secrets
    cfg = get_rag_config()

    # 2. Harmless sample string
    sample_text = "NVIDIA NIM Nemotron embedding verification test."

    print("==================================================")
    print("NVIDIA NIM EMBEDDING STAGE VERIFICATION")
    print("==================================================")
    print(f"[*] Target Model    : {cfg.embed_model}")
    print(f"[*] Endpoint URL    : {cfg.nvidia_base_url}")
    print(f"[*] Sample Input    : \"{sample_text}\"")
    print(f"[*] API Key Present : {'Yes (Protected)' if bool(cfg.nvidia_api_key) else 'No'}")
    print("--------------------------------------------------")

    # 3. Embed the text and measure vector length
    vector = embed_text(sample_text, config=cfg, input_type="query")
    vector_length = len(vector)

    print(f"[OK] Embedding generated successfully!")
    print(f"[*] Returned Vector Length : {vector_length} dimensions")
    print(f"[*] Expected Vector Length : {EXPECTED_DIMENSION} dimensions")
    print(f"[*] Dimension Verified     : {vector_length == EXPECTED_DIMENSION}")
    print("==================================================")


if __name__ == "__main__":
    main()
