"""Process-local Mem0 providers; real cached E5, deliberately disabled inference.

Machine: A-2022BAYAREA (Хаб) · Agent: Codex · Operator: Anton · 2026-11-03
Rail: local CPU model; no network/API. Only the runner config selects these adapters.
The SDK's provider aliases are overridden in this process, never on disk.
"""
from pathlib import Path

from mem0.embeddings.base import EmbeddingBase
from mem0.llms.base import LLMBase

E5_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
E5_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-base/snapshots" / E5_REVISION)


class LocalE5(EmbeddingBase):
    """Real E5 encoder with the passage/query prefixes expected by the model."""
    def __init__(self, config=None):
        super().__init__(config)
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        self.model = SentenceTransformer(str(E5_SNAPSHOT), device="cpu", local_files_only=True)
        self.calls = 0
        self.config.embedding_dims = 768

    def embed(self, text, memory_action=None):
        self.calls += 1
        prefix = "query: " if memory_action == "search" else "passage: "
        return self.model.encode([prefix + text], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)[0].tolist()

    def embed_batch(self, texts, memory_action="add"):
        if not texts:
            return []
        self.calls += len(texts)
        prefix = "query: " if memory_action == "search" else "passage: "
        return self.model.encode([prefix + text for text in texts], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()


class DisabledLLM(LLMBase):
    """Any unexpected extraction call fails instead of quietly using a paid API."""
    calls = 0

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        type(self).calls += 1
        raise RuntimeError("Inference is disabled in this retrieval-only configuration")


def configure_local_providers():
    from mem0.utils.factory import EmbedderFactory, LlmFactory
    # Mem0 validates provider names separately; retain a supported alias while
    # recording the exact implementation in every result. No SDK source edits.
    EmbedderFactory.provider_to_class["huggingface"] = "local_providers.LocalE5"
    LlmFactory.register_provider("openai", "local_providers.DisabledLLM")
