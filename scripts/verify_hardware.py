"""Day 1 — Hardware verification.

Confirms that PyTorch can use the available accelerator (MPS on Apple Silicon,
CUDA on NVIDIA GPUs) or falls back to CPU. Also runs a tiny embedding test
to confirm the full stack is functional end-to-end.
"""
import time
import torch
from sentence_transformers import SentenceTransformer


def detect_device() -> str:
    """Detect best available device for PyTorch."""
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ MPS available (Apple Silicon)")
    else:
        device = "cpu"
        print("⚠️  No accelerator found, falling back to CPU")
    return device


def smoke_test(device: str) -> None:
    """Load a tiny sentence-transformer and embed a few sentences."""
    print(f"\nLoading 'all-MiniLM-L6-v2' on {device}...")
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    load_time = time.time() - t0
    print(f"   Loaded in {load_time:.1f}s")

    sentences = [
        "Apple reported revenue of $394.3B for FY2022.",
        "Microsoft's cloud business grew 28% year over year.",
        "The 10-K filing discusses material risk factors.",
    ]
    t0 = time.time()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    embed_time = time.time() - t0

    print(f"   Embedded {len(sentences)} sentences in {embed_time*1000:.0f}ms")
    print(f"   Embedding shape: {tuple(embeddings.shape)}")
    print(f"   Embedding dtype: {embeddings.dtype}")
    print(f"   Embedding device: {embeddings.device}")

    # Cosine similarity sanity check
    from torch.nn.functional import cosine_similarity
    sim_01 = cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    sim_02 = cosine_similarity(embeddings[0:1], embeddings[2:3]).item()
    print(f"\n   Cosine sim (Apple vs Microsoft revenue): {sim_01:.3f}")
    print(f"   Cosine sim (Apple revenue vs 10-K filing): {sim_02:.3f}")
    print("   ✅ Smoke test passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 1 — Hardware verification for financebench-rag-eval")
    print("=" * 60)
    device = detect_device()
    smoke_test(device)
    print("\n🎉 Setup complete. Ready for Day 2.\n")
