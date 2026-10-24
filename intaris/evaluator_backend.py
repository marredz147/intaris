"""Per-call evaluator backend selection.

``EVALUATOR_BACKEND`` is ``llm``, ``jev``, ``decisions`` or ``auto``. ``auto`` is only valid
with Cognis-managed inference: the evaluation route's ``GET /v1/models`` entry
lists the operations it resolves (``chat`` or ``systemone``), and that decides
whether the chat LLM evaluator or the Jev evaluator handles a call. The lookup
is process-local, cached and coalesced so evaluations never pay for it
individually.

Cognis rejects a protocol mismatch (for example a chat request to a Jev route)
before upstream dispatch with HTTP 400 ``invalid_model``. Both evaluator
clients report that rejection here. The current evaluation fails closed; in
``auto`` mode the cache is cleared so the next call uses the new backend.
Nothing is replayed and nothing is inferred from model names.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from intaris.finite_questions import FiniteQuestionClient

logger = logging.getLogger(__name__)

EVALUATION_ROUTE = "routes/intaris_evaluation"
MODES = ("llm", "jev", "auto", "decisions")
CACHE_SECONDS = 60.0
# Transport failures are retried sooner, but still coalesced and bounded.
FAILURE_CACHE_SECONDS = 5.0


class EvaluatorBackendError(RuntimeError):
    """The evaluator backend is unknown or mismatched; the evaluation fails closed."""


class EvaluatorBackendUnavailable(EvaluatorBackendError):
    """Transient metadata failure; clients must wait for the cache cooldown."""

    def __init__(self, message: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def is_protocol_mismatch(status: int | None, body: Any) -> bool:
    """Return True for Cognis's pre-dispatch ``invalid_model`` rejection."""
    if status != 400 or not isinstance(body, dict):
        return False
    error = body.get("error", body)
    return isinstance(error, dict) and error.get("code") == "invalid_model"


class EvaluatorBackend:
    """Select the evaluator backend for each call.

    ``select()`` returns a finite client when Jev or Decisions should evaluate and
    ``None`` when the chat LLM evaluator should. It raises
    ``EvaluatorBackendError`` when the backend cannot be determined safely.
    """

    def __init__(
        self,
        mode: str,
        *,
        jev_factory: Callable[[], FiniteQuestionClient] | None = None,
        decisions_factory: Callable[[], FiniteQuestionClient] | None = None,
        models_url: str | None = None,
        token: Callable[[], str] | None = None,
        timeout_s: float = 4.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unsupported evaluator backend {mode!r}")
        if mode == "auto" and (models_url is None or token is None):
            raise ValueError("EVALUATOR_BACKEND=auto requires managed inference")
        if mode in ("jev", "auto") and jev_factory is None:
            raise ValueError(f"EVALUATOR_BACKEND={mode} requires a Jev client")
        if mode == "decisions" and decisions_factory is None:
            raise ValueError("EVALUATOR_BACKEND=decisions requires a Decisions client")
        self.mode = mode
        self._jev_factory = decisions_factory if mode == "decisions" else jev_factory
        self._jev: FiniteQuestionClient | None = None
        self._models_url = models_url
        self._token = token
        self._clock = clock
        self._lock = threading.Lock()
        self._jev_lock = threading.Lock()
        self._expires = 0.0
        self._backend: str | None = None
        self._error: EvaluatorBackendError | None = None
        self._absent_logged = False
        self._lookups = 0
        self._generation = 0
        self._pending: Future[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        # Latest pinned-mode mismatch, kept for diagnostics only (never switches).
        self._pin_mismatch: tuple[float, str] | None = None
        self._http = (
            httpx.Client(timeout=timeout_s, transport=transport)
            if mode == "auto"
            else None
        )

    @classmethod
    def fixed(cls, jev: FiniteQuestionClient | None) -> EvaluatorBackend:
        """Pin a pre-built client (or the LLM evaluator when ``None``)."""
        if jev is None:
            return cls("llm")
        if getattr(jev, "provider", None) == "decisions":
            return cls("decisions", decisions_factory=lambda: jev)
        return cls("jev", jev_factory=lambda: jev)

    @property
    def lookups(self) -> int:
        """Number of model-metadata lookups performed (for tests and metrics)."""
        return self._lookups

    def select(self) -> FiniteQuestionClient | None:
        """Return the finite client, or ``None`` for the chat evaluator."""
        with self._lock:
            if self._closed:
                raise EvaluatorBackendError("Evaluator backend is closed")
        backend = self.mode if self.mode != "auto" else self._resolve()
        return self._jev_client() if backend in ("jev", "decisions") else None

    async def select_async(self) -> FiniteQuestionClient | None:
        """Select without occupying a request worker while metadata is pending."""
        if self.mode != "auto":
            return self.select()
        while True:
            backend, pending = self._resolution()
            if pending is None:
                return self._jev_client() if backend == "jev" else None
            await asyncio.shield(asyncio.wrap_future(pending))

    def active(self) -> dict[str, Any]:
        """Describe the backend for diagnostics without raising.

        A pinned backend that Cognis rejected within the last cache window is
        reported with that error, while ``active`` stays the pinned mode.
        """
        if self.mode != "auto":
            mismatch = self._pin_mismatch
            recent = (
                mismatch is not None and self._clock() - mismatch[0] < CACHE_SECONDS
            )
            return {
                "mode": self.mode,
                "active": self.mode,
                "error": mismatch[1] if recent else None,
            }
        try:
            backend = self._resolve()
        except EvaluatorBackendError as exc:
            return {"mode": self.mode, "active": None, "error": str(exc)}
        return {"mode": self.mode, "active": backend, "error": None}

    def protocol_mismatch(self, protocol: str) -> EvaluatorBackendError:
        """Handle Cognis's pre-dispatch rejection of ``protocol`` for this route."""
        if self.mode != "auto":
            logger.error(
                "Cognis rejected the %s protocol for %s while EVALUATOR_BACKEND=%s "
                "is pinned; evaluation failed closed",
                protocol,
                EVALUATION_ROUTE,
                self.mode,
            )
            error = EvaluatorBackendError(
                f"EVALUATOR_BACKEND={self.mode} is pinned, but Cognis rejected the "
                f"{protocol} protocol for {EVALUATION_ROUTE}. Select a matching "
                "model in Cognis routing or use EVALUATOR_BACKEND=auto."
            )
            self._pin_mismatch = (self._clock(), str(error))
            return error
        with self._lock:
            self._generation += 1
            self._expires = 0.0
            self._backend = None
            self._error = None
        logger.warning(
            "Cognis rejected the %s protocol for %s; evaluation failed closed and "
            "the evaluator backend will be refreshed on the next call",
            protocol,
            EVALUATION_ROUTE,
        )
        return EvaluatorBackendError(
            f"Cognis rejected the {protocol} protocol for {EVALUATION_ROUTE}; the "
            "evaluation route changed. This evaluation failed closed; the next "
            "call uses the refreshed backend."
        )

    def close(self) -> None:
        """Release HTTP resources held by the selector and the Jev client."""
        with self._lock:
            self._closed = True
            executor = self._executor
        # Drain before closing HTTP. Its existing configured timeout still applies.
        if executor is not None:
            executor.shutdown(wait=True)
        if self._http is not None:
            self._http.close()
        with self._jev_lock:
            if self._jev is not None:
                self._jev.close()

    async def aclose(self) -> None:
        """Drain the single metadata worker without blocking the event loop."""
        await asyncio.to_thread(self.close)

    def _jev_client(self) -> FiniteQuestionClient:
        assert self._jev_factory is not None
        with self._jev_lock:
            with self._lock:
                if self._closed:
                    raise EvaluatorBackendError("Evaluator backend is closed")
            if self._jev is None:
                self._jev = self._jev_factory()
            return self._jev

    def _resolve(self) -> str:
        while True:
            backend, pending = self._resolution()
            if pending is None:
                assert backend is not None
                return backend
            pending.result()

    def _resolution(self) -> tuple[str | None, Future[None] | None]:
        """Read cache state or join/start exactly one shared refresh."""
        with self._lock:
            if self._closed:
                raise EvaluatorBackendError("Evaluator backend is closed")
            if self._clock() >= self._expires:
                if self._pending is None or self._pending.done():
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(
                            max_workers=1, thread_name_prefix="intaris-metadata"
                        )
                    self._pending = self._executor.submit(
                        self._refresh, self._generation
                    )
                return None, self._pending
            if self._error is not None:
                if isinstance(self._error, EvaluatorBackendUnavailable):
                    # Fresh exception per request avoids accumulating tracebacks.
                    raise EvaluatorBackendUnavailable(
                        str(self._error),
                        max(1, math.ceil(self._expires - self._clock())),
                    )
                raise self._error
            assert self._backend is not None
            return self._backend, None

    def _refresh(self, generation: int) -> None:
        with self._lock:
            self._lookups += 1
        backend, error = None, None
        ttl = CACHE_SECONDS
        try:
            payload = self._fetch()
        except EvaluatorBackendError as exc:
            error = exc
            ttl = FAILURE_CACHE_SECONDS
            logger.warning("Evaluator backend lookup failed: %s", exc)
        else:
            try:
                backend = self._parse(payload)
            except EvaluatorBackendError as exc:
                error = exc
                logger.error("Evaluator backend metadata rejected: %s", exc)
            else:
                logger.info("Evaluator backend selected: %s", backend)
        with self._lock:
            if generation == self._generation and not self._closed:
                self._backend, self._error = backend, error
                self._expires = self._clock() + ttl

    def _fetch(self) -> Any:
        assert self._http is not None and self._token is not None
        try:
            token = self._token()
            response = self._http.get(
                self._models_url, headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError as exc:
            raise EvaluatorBackendUnavailable(
                f"Cognis model metadata is unavailable ({type(exc).__name__})",
                math.ceil(FAILURE_CACHE_SECONDS),
            ) from None
        except RuntimeError as exc:
            raise EvaluatorBackendError(
                f"Cognis model metadata is unavailable ({type(exc).__name__})"
            ) from None
        if response.status_code in {429, 500, 502, 503, 504}:
            raise EvaluatorBackendUnavailable(
                f"Cognis model metadata request failed with HTTP {response.status_code}",
                math.ceil(FAILURE_CACHE_SECONDS),
            )
        if response.status_code != 200:
            raise EvaluatorBackendError(
                f"Cognis model metadata request failed with HTTP {response.status_code}"
            )
        try:
            return response.json()
        except ValueError:
            raise EvaluatorBackendError("Cognis model metadata is not JSON") from None

    def _parse(self, payload: Any) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise EvaluatorBackendError("Cognis model list is malformed")
        entry = next(
            (
                item
                for item in data
                if isinstance(item, dict) and item.get("id") == EVALUATION_ROUTE
            ),
            None,
        )
        if entry is None:
            raise EvaluatorBackendError(
                f"{EVALUATION_ROUTE} is not available to Intaris in Cognis"
            )
        if "cognis" not in entry:
            if not self._absent_logged:
                self._absent_logged = True
                logger.warning(
                    "Cognis does not report operations for %s (older Cognis); "
                    "using the LLM evaluator",
                    EVALUATION_ROUTE,
                )
            return "llm"
        metadata = entry["cognis"]
        operations = metadata.get("operations") if isinstance(metadata, dict) else None
        if not isinstance(operations, list) or not all(
            isinstance(item, str) for item in operations
        ):
            raise EvaluatorBackendError(
                f"Cognis operations metadata for {EVALUATION_ROUTE} is malformed"
            )
        if set(operations) == {"systemone"}:
            return "jev"
        if set(operations) == {"chat"}:
            return "llm"
        raise EvaluatorBackendError(
            f"Cognis reports unsupported operations {sorted(set(operations))} "
            f"for {EVALUATION_ROUTE}"
        )
