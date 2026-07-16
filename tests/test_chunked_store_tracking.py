import gc
import importlib.util
from pathlib import Path
import textwrap
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "base/patches/patch_chunked_store_futures.py"
PROMPT_TOKENS = 258_048
STORE_STEP_TOKENS = 3_072


def load_patcher():
    spec = importlib.util.spec_from_file_location("chunked_store_patcher", PATCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def __init__(self) -> None:
        self.errors: list[tuple[object, ...]] = []

    def error(self, *args: object) -> None:
        self.errors.append(args)


def load_poll_method():
    patcher = load_patcher()
    namespace = {"logger": FakeLogger()}
    exec(textwrap.dedent(patcher.build_poll_method()), namespace)
    return namespace["_poll_store_futures"]


class FakeFuture:
    def __init__(self, chunk: int) -> None:
        self.chunk = chunk
        self.done = False
        self.result_calls = 0

    def query(self) -> bool:
        return self.done

    def result(self) -> bool:
        if not self.done:
            raise RuntimeError("result called before completion")
        self.result_calls += 1
        return True


class FakeEvent:
    def __init__(self, chunk: int) -> None:
        self.chunk = chunk


class Adapter:
    _poll_store_futures = load_poll_method()

    def __init__(self) -> None:
        self.store_futures: dict[str, list[FakeFuture]] = {}
        self.store_events: dict[str, list[FakeEvent]] = {}


class ChunkedStoreTrackingTests(unittest.TestCase):
    def test_exact_258048_token_failure_shape(self) -> None:
        chunk_count, remainder = divmod(PROMPT_TOKENS, STORE_STEP_TOKENS)
        self.assertEqual(remainder, 0)
        self.assertEqual(chunk_count, 84)

        adapter = Adapter()
        futures = [FakeFuture(chunk) for chunk in range(chunk_count)]
        events = [FakeEvent(chunk) for chunk in range(chunk_count)]
        event_refs = [weakref.ref(event) for event in events]
        adapter.store_futures["request"] = list(futures)
        adapter.store_events["request"] = list(events)
        del events
        gc.collect()

        self.assertTrue(all(ref() is not None for ref in event_refs))

        # Complete alternating chunks to ensure the implementation preserves
        # future/event alignment rather than only tracking a count.
        for future in futures[::2]:
            future.done = True
        self.assertEqual(adapter._poll_store_futures(), set())
        self.assertEqual(adapter.store_futures["request"], futures[1::2])
        self.assertEqual(
            [event.chunk for event in adapter.store_events["request"]],
            list(range(1, chunk_count, 2)),
        )

        gc.collect()
        self.assertTrue(
            all(event_refs[index]() is None for index in range(0, chunk_count, 2))
        )
        self.assertTrue(
            all(event_refs[index]() is not None for index in range(1, chunk_count, 2))
        )

        # Keep the 84th chunk pending across a second poll.
        for future in futures[1:-1:2]:
            future.done = True
        self.assertEqual(adapter._poll_store_futures(), set())
        self.assertEqual(adapter.store_futures["request"], [futures[-1]])
        self.assertEqual(adapter.store_events["request"][0].chunk, chunk_count - 1)

        futures[-1].done = True
        self.assertEqual(adapter._poll_store_futures(), {"request"})
        self.assertEqual(adapter.store_futures, {})
        self.assertEqual(adapter.store_events, {})
        self.assertTrue(all(future.result_calls == 1 for future in futures))

        gc.collect()
        self.assertTrue(all(ref() is None for ref in event_refs))

    def test_scalar_bookkeeping_reproduces_lifetime_loss(self) -> None:
        scalar_events: dict[str, FakeEvent] = {}
        refs: list[weakref.ReferenceType[FakeEvent]] = []
        for chunk in range(84):
            event = FakeEvent(chunk)
            refs.append(weakref.ref(event))
            scalar_events["request"] = event
        del event
        gc.collect()

        self.assertEqual(sum(ref() is not None for ref in refs), 1)
        self.assertEqual(scalar_events["request"].chunk, 83)

    def test_misaligned_state_fails_closed(self) -> None:
        adapter = Adapter()
        adapter.store_futures["request"] = [FakeFuture(0)]
        adapter.store_events["request"] = []
        with self.assertRaisesRegex(RuntimeError, "future/event mismatch"):
            adapter._poll_store_futures()


if __name__ == "__main__":
    unittest.main()

