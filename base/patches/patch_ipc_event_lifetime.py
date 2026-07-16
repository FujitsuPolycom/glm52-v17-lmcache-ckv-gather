#!/usr/bin/env python3
"""Backport LMCache #3245: retain CUDA IPC events until transfers finish."""

import os
from pathlib import Path


ADAPTER_PATH = Path(
    os.environ.get(
        "LMCACHE_ADAPTER_PATH",
        "/opt/venv/lib/python3.12/site-packages/"
        "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    )
)
MARKER = "AI01-LMCACHE-IPC-EVENT-LIFETIME"


def replace_once(content: str, old: str, new: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match, found {count}: {old[:80]!r}")
    return content.replace(old, new, 1)


def main() -> None:
    content = ADAPTER_PATH.read_text()
    if MARKER in content:
        print(f"[{MARKER}] Already patched, skipping.")
        return

    content = replace_once(
        content,
        "from typing import Any, Callable, NoReturn",
        "from typing import Any, Callable, NoReturn, Protocol",
    )
    content = replace_once(
        content,
        "\ndef wrap_kv_caches(",
        f'''\n# {MARKER}: backport LMCache/LMCache#3245.\nclass _IpcEvent(Protocol):\n    def ipc_handle(self) -> Any: ...\n\n\ndef wrap_kv_caches(''',
    )
    content = replace_once(
        content,
        '''        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}

        # Block IDs that failed due to retrieve timeout''',
        '''        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[int]]
        ] = {}
        # A CUDA IPC handle is not sufficient by itself. Keep the exporting
        # event alive until the receiving process finishes the transfer.
        self.store_events: dict[str, _IpcEvent] = {}
        self.retrieve_events: dict[str, _IpcEvent] = {}

        # Block IDs that failed due to retrieve timeout''',
    )
    content = replace_once(
        content,
        '''        )
        self.store_futures[request_id] = future

    @_lmcache_nvtx_annotate
    def submit_retrieve_request''',
        '''        )
        self.store_futures[request_id] = future
        self.store_events[request_id] = event

    @_lmcache_nvtx_annotate
    def submit_retrieve_request''',
    )
    content = replace_once(
        content,
        '''        )
        self.retrieve_futures[request_id] = (future, list(op.block_ids))

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests''',
        '''        )
        self.retrieve_futures[request_id] = (future, list(op.block_ids))
        self.retrieve_events[request_id] = event

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests''',
    )
    content = replace_once(
        content,
        '''            self.store_futures.clear()
            self.retrieve_futures.clear()

            ret_stores = self._process_finished_stores''',
        '''            self.store_futures.clear()
            self.retrieve_futures.clear()
            self.store_events.clear()
            self.retrieve_events.clear()

            ret_stores = self._process_finished_stores''',
    )
    content = replace_once(
        content,
        '''        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)''',
        '''        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
            self.store_events.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)
            self.retrieve_events.pop(request_id, None)''',
    )

    # These annotations are documentation and catch accidental non-IPC callers.
    content = content.replace("        event: Any,", "        event: _IpcEvent,")
    if content.count("event: _IpcEvent,") != 4:
        raise RuntimeError("expected four IPC event annotations")

    ADAPTER_PATH.write_text(content)
    print(f"[{MARKER}] Applied LMCache/LMCache commit 5824ab308906.")


if __name__ == "__main__":
    main()
