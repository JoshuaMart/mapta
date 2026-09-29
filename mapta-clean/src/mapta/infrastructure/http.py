"""Shared HTTP client lifecycle for the adapters that speak to a web API."""

# httpx2 is httpx 2.x under its new name; it is what the openai SDK expects.
import httpx2 as httpx

__all__ = ["HTTPAdapter"]

DEFAULT_TIMEOUT = 30.0


class HTTPAdapter:
    """Base for adapters that own an ``AsyncClient`` unless one is injected.

    An injected client belongs to the caller -- a test's ``MockTransport``, or a
    shared pool -- so ``aclose`` must leave it alone.
    """

    def __init__(
        self, client: httpx.AsyncClient | None = None, *, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
