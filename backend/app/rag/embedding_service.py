from typing import Sequence


class EmbeddingService:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[] for _ in texts]
