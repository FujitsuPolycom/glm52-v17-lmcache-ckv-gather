#!/usr/bin/env python3
"""Retain every outstanding LMCache store future for chunked prefill."""

import os
from pathlib import Path


ADAPTER_PATH = Path(
    os.environ.get(
        "LMCACHE_ADAPTER_PATH",
        "/opt/venv/lib/python3.12/site-packages/"
        "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    )
)
MARKER = "AI01-LMCACHE-CHUNKED-STORE-FUTURES"


def replace_once(content: str, old: str, new: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match, found {count}: {old[:80]!r}")
    return content.replace(old, new, 1)


def build_poll_method() -> str:
    """Return the exact method injected into LMCache's worker adapter."""
    return f'''    # {MARKER}: drain each chunk future with its matching IPC event.
    def _poll_store_futures(self) -> set[str]:
        finished_requests: set[str] = set()
        for request_id, futures in list(self.store_futures.items()):
            events = self.store_events.get(request_id, [])
            if len(futures) != len(events):
                raise RuntimeError(
                    f"LMCache store future/event mismatch for {{request_id}}: "
                    f"{{len(futures)}} futures vs {{len(events)}} events"
                )

            pending_futures: list[MessagingFuture[StoreResult]] = []
            pending_events: list[_IpcEvent] = []
            for future, event in zip(futures, events, strict=True):
                if not future.query():
                    pending_futures.append(future)
                    pending_events.append(event)
                    continue

                if not future.result():
                    logger.error(
                        "Something went wrong when processing the "
                        "store request for request_id=%s",
                        request_id,
                    )

            if pending_futures:
                self.store_futures[request_id] = pending_futures
                self.store_events[request_id] = pending_events
            else:
                self.store_futures.pop(request_id, None)
                self.store_events.pop(request_id, None)
                finished_requests.add(request_id)

        return finished_requests

'''


def main() -> None:
    content = ADAPTER_PATH.read_text()
    if MARKER in content:
        print(f"[{MARKER}] Already patched, skipping.")
        return
    if "AI01-LMCACHE-IPC-EVENT-LIFETIME" not in content:
        raise RuntimeError("CUDA IPC event-lifetime backport must be applied first")

    content = replace_once(
        content,
        "self.store_futures: dict[str, MessagingFuture[StoreResult]] = {}",
        "self.store_futures: dict[str, list[MessagingFuture[StoreResult]]] = {}",
    )
    content = replace_once(
        content,
        "self.store_events: dict[str, _IpcEvent] = {}",
        "self.store_events: dict[str, list[_IpcEvent]] = {}",
    )
    content = replace_once(
        content,
        '''        self.store_futures[request_id] = future
        self.store_events[request_id] = event''',
        '''        # A request can submit one store per chunked-prefill step. Do not
        # overwrite an older in-flight CUDA IPC future or its exporting event.
        self.finished_stores.discard(request_id)
        self.store_futures.setdefault(request_id, []).append(future)
        self.store_events.setdefault(request_id, []).append(event)''',
    )

    poll_method = build_poll_method()
    content = replace_once(
        content,
        "    def _process_finished_stores(\n",
        poll_method + "    def _process_finished_stores(\n",
    )

    old_loop = '''        finished_stores = set()
        finished_retrieves = set()
        for request_id, s_future in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)

            if not s_result:
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, _) in self.retrieve_futures.items():'''
    new_loop = '''        finished_stores = self._poll_store_futures()
        finished_retrieves = set()
        for request_id, (r_future, _) in self.retrieve_futures.items():'''
    content = replace_once(content, old_loop, new_loop)

    content = replace_once(
        content,
        '''        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
            self.store_events.pop(request_id, None)
        for request_id in finished_retrieves:''',
        '''        # Store futures/events are removed together by _poll_store_futures.
        for request_id in finished_retrieves:''',
    )

    ADAPTER_PATH.write_text(content)
    print(f"[{MARKER}] Applied per-request chunk future/event tracking.")


if __name__ == "__main__":
    main()
