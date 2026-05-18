from app.core.config import Settings
from app.rag.embeddings import SentenceTransformerEmbeddingProvider, build_embedding_provider


class FakeSentenceTransformer:
    instances = []

    def __init__(
        self,
        model_name: str,
        device: str,
        cache_folder: str | None,
        trust_remote_code: bool,
        local_files_only: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.calls: list[list[str]] = []
        FakeSentenceTransformer.instances.append(self)

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_sentence_transformer_provider_applies_e5_prefixes(monkeypatch) -> None:
    FakeSentenceTransformer.instances.clear()
    monkeypatch.setattr("app.rag.embeddings._load_sentence_transformer", lambda: FakeSentenceTransformer)

    provider = SentenceTransformerEmbeddingProvider(
        Settings(
            embedding_provider="sentence_transformers",
            embedding_model="intfloat/multilingual-e5-small",
            embedding_dimensions=384,
            embedding_device="cpu",
            embedding_batch_size=4,
            embedding_cache_path="./cache-for-test",
            embedding_local_files_only=True,
        )
    )

    query_vector = provider.embed_query("如何部署服务？")
    passage_vectors = provider.embed_passages(["先配置环境变量。"])

    instance = FakeSentenceTransformer.instances[0]
    assert instance.cache_folder == "./cache-for-test"
    assert instance.local_files_only is True
    assert instance.calls[0] == ["query: 如何部署服务？"]
    assert instance.calls[1] == ["passage: 先配置环境变量。"]
    assert query_vector == [1.0, 0.0, 0.0]
    assert passage_vectors == [[1.0, 0.0, 0.0]]


def test_build_embedding_provider_supports_sentence_transformers(monkeypatch) -> None:
    FakeSentenceTransformer.instances.clear()
    monkeypatch.setattr("app.rag.embeddings._load_sentence_transformer", lambda: FakeSentenceTransformer)

    provider = build_embedding_provider(
        Settings(
            embedding_provider="sentence_transformers",
            embedding_model="intfloat/multilingual-e5-small",
            embedding_dimensions=384,
        )
    )

    vector = provider.embed_query("部署步骤")
    assert vector == [1.0, 0.0, 0.0]
