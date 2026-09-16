"""
The one place the model is actually called.

Everything else in `agents/` talks to the model through `complete()`. That matters for a
reason that has nothing to do with tidiness: an unguarded
`client.chat.completions.create` raises straight through LangGraph and takes the whole
run down. No decision, no record, no outcome -- just a traceback, on a demo night, in
front of the people grading it.

A rate limit is not a reason to lose a safety finding. Transient failures are retried
with backoff; a persistent one raises `ModelUnavailable`, which the agents turn into a
parked case with `Blocker.MODEL_UNAVAILABLE`. The event is still on file and a human
still sees it in the queue. That is the difference between degraded and lost.

Retries are deliberately few. This is not a high-throughput service, and a finding that
takes thirty seconds to assess is fine; one that silently burns budget against a broken
key is not.
"""

from __future__ import annotations

import time

ATTEMPTS = 3
BASE_DELAY = 1.0        # seconds; doubles each retry. Tests set this to 0.

# Matched against the exception's class name and message, because the OpenAI SDK's
# exception types have moved between versions and a hard import would tie this file
# to one of them.
TRANSIENT = ("ratelimit", "rate limit", "timeout", "timedout", "connection",
             "apiconnection", "overloaded", "service unavailable", "internalserver",
             "502", "503", "504", "429")

FATAL = ("authentication", "permission", "invalid_api_key", "invalid api key",
         "notfound", "not found", "badrequest", "invalid request", "401", "403", "404")


class ModelUnavailable(RuntimeError):
    """The model could not be reached, after retrying what was worth retrying."""


def _is_transient(exc: Exception) -> bool:
    """A wrong key will still be wrong in two seconds. Only retry what might change."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(word in text for word in FATAL):
        return False
    return any(word in text for word in TRANSIENT)


def complete(client, **kwargs):
    """`client.chat.completions.create(**kwargs)`, with retries on transient failure.

    Raises ModelUnavailable rather than the SDK's own exception, so callers do not have
    to know which SDK version they are on to catch it.
    """
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:               # noqa: BLE001 -- see TRANSIENT above
            last = exc
            if not _is_transient(exc) or attempt == ATTEMPTS - 1:
                break
            time.sleep(BASE_DELAY * (2 ** attempt))
    raise ModelUnavailable(f"{type(last).__name__}: {last}") from last
