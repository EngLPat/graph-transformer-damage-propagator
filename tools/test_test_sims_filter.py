#!/usr/bin/env python3
"""Unit tests for the GTP_TEST_SIMS narrowing helpers in src/gtp/__main__.py.

The helpers decide WHICH simulations a rollout covers. A silent mistake there
would render or score the wrong simulation while every log line still looked
right, so the contract is tested directly rather than inferred from a run.

Run: python3 tools/test_test_sims_filter.py
"""
import os, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.gtp.__main__ import (parse_test_sims_spec, narrow_test_split_to_sims,
                              sim_id_of)

FAILURES = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILURES.append(name)


def expect_raises(name, fn, *a):
    try:
        fn(*a)
    except ValueError:
        check(name, True)
        return
    except Exception as e:
        check(name + f" (raised {type(e).__name__}, wanted ValueError)", False)
        return
    check(name + " (no exception raised)", False)


class Frame:
    """Stand-in for a PyG Data transition carrying sample_idx and a time order."""
    def __init__(self, sid, t):
        self.sample_idx = types.SimpleNamespace(item=lambda s=sid: s)
        self.t = t

    def __repr__(self):
        return f"F(sim={sim_id_of(self)},t={self.t})"


def main():
    print("\n--- parse_test_sims_spec ---")
    check("unset -> None (no-op)", parse_test_sims_spec(None) is None)
    check("empty -> None", parse_test_sims_spec("") is None)
    check("whitespace -> None", parse_test_sims_spec("   ") is None)
    check("single id", parse_test_sims_spec("13") == [13])
    check("comma separated", parse_test_sims_spec("13,22") == [13, 22])
    check("space separated", parse_test_sims_spec("13 22") == [13, 22])
    check("mixed + unsorted + dup",
          parse_test_sims_spec("22, 13 13") == [13, 22])
    expect_raises("non-integer token rejected", parse_test_sims_spec, "13,abc")
    expect_raises("float token rejected", parse_test_sims_spec, "13.5")

    print("\n--- narrow_test_split_to_sims ---")
    # A time-ordered split holding three simulations, interleaved in id order
    # but contiguous per simulation, exactly as the real split is built.
    split = ([Frame(7, t) for t in range(4)]
             + [Frame(13, t) for t in range(5)]
             + [Frame(22, t) for t in range(3)])

    check("None keep -> identical content",
          narrow_test_split_to_sims(split, None) == split)
    check("empty keep -> identical content",
          narrow_test_split_to_sims(split, []) == split)
    check("None keep returns a COPY, not the same list object",
          narrow_test_split_to_sims(split, None) is not split)

    got = narrow_test_split_to_sims(split, [13])
    check("single sim: right count", len(got) == 5)
    check("single sim: right id", {sim_id_of(d) for d in got} == {13})
    check("single sim: time order preserved",
          [d.t for d in got] == [0, 1, 2, 3, 4])

    got2 = narrow_test_split_to_sims(split, [22, 7])
    check("two sims: right count", len(got2) == 7)
    check("two sims: original relative order preserved (7 before 22)",
          [sim_id_of(d) for d in got2] == [7, 7, 7, 7, 22, 22, 22])

    check("input list not mutated", len(split) == 12)

    expect_raises("absent sim rejected", narrow_test_split_to_sims, split, [99])
    expect_raises("partially absent set rejected",
                  narrow_test_split_to_sims, split, [13, 99])

    print("\n--- sim_id_of ---")
    check("reads sample_idx via .item()", sim_id_of(Frame(13, 0)) == 13)
    check("plain int sample_idx accepted",
          sim_id_of(types.SimpleNamespace(sample_idx=13)) == 13)
    check("missing sample_idx -> None",
          sim_id_of(types.SimpleNamespace()) is None)

    print("\n" + ("ALL TESTS PASSED" if not FAILURES
                  else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
