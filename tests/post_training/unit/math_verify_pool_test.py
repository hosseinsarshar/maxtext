# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for math_verify_pool grading and score-assignment logic."""
import logging
import os
import subprocess
import sys
import pytest
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy

from maxtext.trainers.post_train.rl import math_verify_pool as mvp
from maxtext.trainers.post_train.rl.math_verify_pool import (
    are_equal_under_sympy,
    math_verify_pool,
    verify_math_worker,
)

pytestmark = [pytest.mark.post_training]


def _make_config(reward=1.0):
  return SimpleNamespace(reward_exact_answer=reward)


class _FakeAsyncResult:
  """Stand-in for multiprocessing.pool.AsyncResult.

  Runs the target synchronously at construction and caches the outcome so
  `ready()`/`get()` match the real `AsyncResult` contract without spawning
  a worker. Lets us drive `math_verify_pool`'s busy-poll in-process.
  """

  def __init__(self, fn, args):
    try:
      self._value = fn(*args)
      self._exc = None
    except Exception as exc:  # pylint: disable=broad-except
      self._value = None
      self._exc = exc

  def ready(self):
    return True

  def get(self, timeout=None):  # pylint: disable=unused-argument
    if self._exc is not None:
      raise self._exc
    return self._value


class _FakePool:
  """Minimal pool stub: runs `apply_async` synchronously in-process."""

  def apply_async(self, fn, args):
    return _FakeAsyncResult(fn, args)


def _fake_get_pool(num_procs):  # pylint: disable=unused-argument
  return _FakePool()


class VerifyMathWorkerTest(unittest.TestCase):
  """Unit tests for the in-process grader (no pool, no spawned workers)."""

  def test_exact_numeric_match(self):
    score = verify_math_worker(["\\boxed{42}"], ["\\boxed{42}"])
    self.assertEqual(score, 1.0)

  def test_numeric_mismatch(self):
    score = verify_math_worker(["\\boxed{42}"], ["\\boxed{99}"])
    self.assertEqual(score, 0.0)

  def test_multiple_golds_one_matches(self):
    score = verify_math_worker(["\\boxed{1}", "\\boxed{42}"], ["\\boxed{42}"])
    self.assertEqual(score, 1.0)

  def test_multiple_golds_none_matches(self):
    score = verify_math_worker(["\\boxed{1}", "\\boxed{2}"], ["\\boxed{99}"])
    self.assertEqual(score, 0.0)

  def test_empty_prediction_returns_zero(self):
    score = verify_math_worker(["\\boxed{42}"], [""])
    self.assertEqual(score, 0.0)

  def test_fraction_equivalent_to_decimal(self):
    # 1/2 and 0.5 are numerically equal — verify() should catch this even
    # if are_equal_under_sympy's structural match does not.
    score = verify_math_worker(["\\boxed{\\frac{1}{2}}"], ["\\boxed{0.5}"])
    self.assertEqual(score, 1.0)


class AreEqualUnderSympyTest(unittest.TestCase):
  """Tests for the structural sympy AST equality helper.

  `are_equal_under_sympy` is invoked first inside the worker and short-circuits
  `verify()` when it returns True. Its job is cheap structural equality on
  unevaluated expressions.
  """

  def test_same_integer(self):
    self.assertTrue(are_equal_under_sympy(sympy.Integer(42), sympy.Integer(42)))

  def test_different_integer(self):
    self.assertFalse(are_equal_under_sympy(sympy.Integer(42), sympy.Integer(99)))

  def test_same_symbol(self):
    x = sympy.Symbol("x")
    self.assertTrue(are_equal_under_sympy(x, x))

  def test_malformed_input_does_not_raise(self):
    # Unparsable strings must not propagate an exception; they return False.
    self.assertFalse(are_equal_under_sympy("$$$", "%%%"))


@patch.object(mvp, "_get_pool", _fake_get_pool)
class MathVerifyPoolScoreAssignmentTest(unittest.TestCase):
  """Regression tests for the score-assignment bug.

  Prior version granted `reward_exact_answer` on every completed job, ignoring
  the grader's score. These tests exist to keep that bug from coming back.

  `_get_pool` is patched with an in-process fake so the busy-poll drains on
  the first iteration — no spawn, no 300s global_timeout.
  """

  def test_correct_answer_gets_reward(self):
    items = [(0, ["\\boxed{42}"], ["\\boxed{42}"])]
    scores = [0.0]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], 1.0)

  def test_wrong_answer_does_not_get_reward(self):
    items = [(0, ["\\boxed{42}"], ["\\boxed{99}"])]
    scores = [0.0]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], 0.0)

  def test_wrong_answer_preserves_prior_penalty(self):
    # `check_numbers` seeds scores[idx] with `penalty_incorrect_format`; a
    # wrong grader verdict must not overwrite that with the reward.
    items = [(0, ["\\boxed{42}"], ["\\boxed{99}"])]
    scores = [-0.5]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], -0.5)

  def test_mixed_batch_scores_each_item_independently(self):
    items = [
        (0, ["\\boxed{1}"], ["\\boxed{1}"]),  # correct
        (1, ["\\boxed{2}"], ["\\boxed{99}"]),  # wrong
        (2, ["\\boxed{3}"], ["\\boxed{3}"]),  # correct
    ]
    scores = [0.0, 0.0, 0.0]
    result = math_verify_pool(_make_config(1.0), items, scores)
    self.assertEqual(result[0], 1.0)
    self.assertEqual(result[1], 0.0)
    self.assertEqual(result[2], 1.0)

  def test_reward_uses_max_not_overwrite(self):
    # A correct answer must not lower an already-higher pre-existing score.
    items = [(0, ["\\boxed{42}"], ["\\boxed{42}"])]
    scores = [0.7]
    result = math_verify_pool(_make_config(0.3), items, scores)
    self.assertEqual(result[0], 0.7)

  def test_empty_items_returns_scores_unchanged(self):
    scores = [0.1, 0.2, 0.3]
    result = math_verify_pool(_make_config(1.0), [], scores)
    self.assertEqual(result, [0.1, 0.2, 0.3])


class _CountingPool(_FakePool):
  """`_FakePool` that records each submitted job's arguments."""

  def __init__(self):
    self.submitted = []

  def apply_async(self, fn, args):
    self.submitted.append(args)
    return super().apply_async(fn, args)


class _NeverReadyResult:

  def ready(self):
    return False


class _FailedResult:

  def ready(self):
    return True

  def get(self, timeout=None):  # pylint: disable=unused-argument
    raise RuntimeError("worker died")


class MathVerifyPoolDedupTest(unittest.TestCase):
  """Items with the same golds and predictions are graded once; each index gets the score."""

  def test_duplicate_items_are_graded_once(self):
    pool = _CountingPool()
    items = [
        (0, ["\\boxed{42}"], ["\\boxed{42}"]),
        (1, ["\\boxed{42}"], ["\\boxed{99}"]),
        (2, ["\\boxed{42}"], ["\\boxed{42}"]),
        (3, ["\\boxed{42}"], ["\\boxed{42}"]),
    ]
    logs = []
    with patch.object(mvp, "_get_pool", return_value=pool):
      result = math_verify_pool(_make_config(1.0), items, [0.0, -0.5, 0.5, 2.0], log_fn=logs.append)
    self.assertEqual(pool.submitted, [(["\\boxed{42}"], ["\\boxed{42}"]), (["\\boxed{42}"], ["\\boxed{99}"])])
    self.assertEqual(result, [1.0, -0.5, 1.0, 2.0])
    self.assertIn("Processed 4/4 items (2 unique, 0 timed out)", logs[-1])

  def test_timed_out_group_keeps_default_scores_and_resets_pool(self):
    pool = _FakePool()
    pool.apply_async = lambda fn, args: _NeverReadyResult()
    items = [(0, ["a"], ["b"]), (1, ["a"], ["b"]), (2, ["a"], ["b"])]
    logs = []
    with patch.object(mvp, "_get_pool", return_value=pool), patch.object(mvp, "_shutdown_pool") as shutdown:
      result = math_verify_pool(_make_config(1.0), items, [-0.5, -0.5, 0.0], timeout=0.05, log_fn=logs.append)
    self.assertEqual(result, [-0.5, -0.5, 0.0])
    shutdown.assert_called_once()
    self.assertIn("Processed 0/3 items (1 unique, 3 timed out)", logs[-1])

  def test_failed_job_keeps_default_scores_for_every_duplicate(self):
    pool = _FakePool()
    pool.apply_async = lambda fn, args: _FailedResult()
    items = [(0, ["a"], ["b"]), (2, ["a"], ["b"])]
    logs = []
    with patch.object(mvp, "_get_pool", return_value=pool):
      result = math_verify_pool(_make_config(1.0), items, [-0.5, 7.0, 0.25], log_fn=logs.append)
    self.assertEqual(result, [-0.5, 7.0, 0.25])
    self.assertIn("failed (worker died) for indices: [0, 2]", logs[0])


class MathVerifyPoolModuleTest(unittest.TestCase):
  """Module-level behavior: import and worker setup."""

  def test_imports_in_a_fresh_interpreter(self):
    # `_get_pool` is annotated with `multiprocessing.pool.Pool`, which needs an
    # explicit `import multiprocessing.pool`. Load the file in a new process so
    # no earlier import can hide a missing one.
    code = (
        "import importlib.util, sys\n"
        "assert 'multiprocessing.pool' not in sys.modules\n"
        f"spec = importlib.util.spec_from_file_location('mvp_fresh', {mvp.__file__!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120)
    self.assertEqual(proc.returncode, 0, proc.stderr)

  def test_worker_init_silences_math_verify_logger(self):
    logger = logging.getLogger("math_verify")
    old_level = logger.level
    try:
      with patch.dict(os.environ):
        mvp.silent_worker_init()
      self.assertEqual(logger.level, logging.CRITICAL)
    finally:
      logger.setLevel(old_level)


if __name__ == "__main__":
  unittest.main()
