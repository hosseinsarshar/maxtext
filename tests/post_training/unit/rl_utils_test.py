# Copyright 2026 Google LLC
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

"""Unit tests for RL result parsing and reward scoring (CPU-only)."""

import os
import pathlib
import re
import tempfile
import unittest
from unittest import mock
import pytest
from types import SimpleNamespace

from maxtext.trainers.post_train.rl import math_verify_pool
from maxtext.trainers.post_train.rl import utils_rl

pytestmark = [pytest.mark.post_training]


def _make_config():
  """Create a minimal config object."""
  return SimpleNamespace(
      reasoning_start_token="<reasoning>",
      reasoning_end_token="</reasoning>",
      solution_start_token="<answer>",
      solution_end_token="</answer>",
      reward_exact_answer=3.0,
      reward_exact_format_match=2.0,
      reward_partial_format_match=0.5,
      reward_white_space_format_match=1.5,
      reward_ratio_guess_to_answer_high=1.0,
      reward_ratio_guess_to_answer_low=0.5,
      penalty_incorrect_format=-0.5,
      penalty_incorrect_answer=-0.5,
      dataset_name="test",
      debug=False,
  )


class TestProcessAnswer(unittest.TestCase):
  """Tests for utils_rl.process_answer."""

  def test_for_mcq(self):
    self.assertEqual(len(utils_rl.process_answer("(A) 1\n(B) 2\n(C) 3\n", "B", "MCQ")), 2)
    self.assertEqual(len(utils_rl.process_answer("A. 1\nB. 2\n(C) 3\n", "B", "MCQ")), 2)
    self.assertEqual(
        len(utils_rl.process_answer("$\\textbf{(A)}~\\frac{1}{24}\\qquad\\textbf{(B)}~\\frac{1}{12}$", "B", "MCQ")), 2
    )
    self.assertEqual(len(utils_rl.process_answer("$(\\mathrm {A}) \\ 1 \\qquad (\\mathrm {B}) \\ 2$", "B", "MCQ")), 2)


class TestNormalizeFinalAnswer(unittest.TestCase):
  """Tests for utils_rl.normalize_final_answer."""

  def test_comma_boxed_and_currency(self):
    # Comma-separated numbers, \\boxed{}, and leading $ are all normalized to plain integers
    self.assertEqual(utils_rl.normalize_final_answer("1,000"), "1000")
    self.assertEqual(utils_rl.normalize_final_answer("$1,000"), "1000")
    self.assertEqual(utils_rl.normalize_final_answer("\\boxed{1,000}"), "1000")

  def test_equation_splitting_and_unit_removal(self):
    # Expressions with '=' are split on '='; trailing unit words are stripped
    self.assertEqual(utils_rl.normalize_final_answer("x = 10"), "10")
    self.assertEqual(utils_rl.normalize_final_answer("total = 100 meters"), "100")
    self.assertEqual(utils_rl.normalize_final_answer("42 mph"), "42")

  def test_latex_wrappers(self):
    # \\text{}, \\textbf{}, and \\overline{} wrappers are removed, leaving inner content
    self.assertEqual(utils_rl.normalize_final_answer("\\text{hello}"), "hello")
    self.assertEqual(utils_rl.normalize_final_answer("\\textbf{42}"), "42")
    self.assertEqual(utils_rl.normalize_final_answer("\\overline{AB}"), "AB")

  def test_dollar_math_extraction(self):
    # Content inside $...$ is extracted
    self.assertEqual(utils_rl.normalize_final_answer("The answer is $\\frac{1}{2}$"), "\\frac{1}{2}")
    self.assertEqual(utils_rl.normalize_final_answer("The answer is 3 $\\frac{1}{2}$"), "3\\frac{1}{2}")

  def test_shorthand_frac_and_sqrt(self):
    # Shorthand \\fracab and \\sqrta are expanded to their full LaTeX forms
    self.assertEqual(utils_rl.normalize_final_answer("\\fracab"), "\\frac{a}{b}")
    self.assertEqual(utils_rl.normalize_final_answer("\\sqrta"), "\\sqrt{a}")


class TestMatchFormatApproximatelyScores(unittest.TestCase):
  """Tests for utils_rl.match_format_approximately.

  Each tag that appears exactly once contributes reward_partial_format_match (0.5).
  Each tag that is absent or appears more than once contributes penalty_incorrect_format (-0.5).
  With 4 tags the score ranges from -2.0 (all wrong) to 2.0 (all correct).
  """

  def setUp(self):
    self.config = _make_config()

  def _score(self, completion):
    return utils_rl.match_format_approximately(None, completion, self.config)

  def test_score_all_tags_present_exactly_once(self):
    # All four tags present exactly once -> 4 * 0.5 = 2.0
    self.assertEqual(self._score(["<reasoning>think</reasoning><answer>42</answer>"])[0], 2.0)

  def test_score_no_tags_present(self):
    # No tags at all -> 4 * -0.5 = -2.0
    self.assertEqual(self._score(["The answer is 42."])[0], -2.0)

  def test_score_only_answer_tags_present(self):
    # Only <answer>...</answer> present -> 2 * 0.5 + 2 * -0.5 = 0.0
    self.assertEqual(self._score(["<answer>42</answer>"])[0], 0.0)

  def test_score_duplicate_reasoning_start_tag(self):
    # Duplicate <reasoning> tag -> 3 * 0.5 + 1 * -0.5 = 1.0
    self.assertEqual(self._score(["<reasoning><reasoning>think</reasoning><answer>42</answer>"])[0], 1.0)

  def test_score_multiple_completions(self):
    # Multiple completions at once -> one score per entry
    multi_completions = [
        "<reasoning>think</reasoning><answer>42</answer>",  # 2.0
        "no tags here",  # -2.0
    ]
    scores = self._score(multi_completions)
    self.assertEqual(len(scores), 2)
    self.assertEqual(scores[0], 2.0)
    self.assertEqual(scores[1], -2.0)

  @pytest.mark.cpu_only
  def test_prefilled_reasoning_start_scores_native_completion(self):
    self.config.reasoning_start_token = "<think>"
    self.config.reasoning_end_token = "</think>"
    self.config.reasoning_start_token_in_prompt = True
    completion = "40 + 2 = 42</think>\n\n<answer>42</answer>"

    self.assertEqual(self._score([completion])[0], 2.0)
    self.assertEqual(
        utils_rl.match_format_exactly(None, [completion], self.config)[0],
        self.config.reward_exact_format_match,
    )

  @pytest.mark.cpu_only
  def test_prefilled_empty_reasoning_block_matches_exact_format(self):
    self.config.reasoning_start_token = "<think>"
    self.config.reasoning_end_token = "</think>"
    self.config.reasoning_start_token_in_prompt = True
    completion = "</think><answer>42</answer>"

    self.assertEqual(
        utils_rl.match_format_exactly(None, [completion], self.config)[0],
        self.config.reward_exact_format_match,
    )


class TestCheckNumbers(unittest.TestCase):
  """Tests for utils_rl.check_numbers.

  Covers two scenarios:
    1. Whether the regex can extract an answer from the completion.
    2. Whether the extracted value matches (or does not match) the reference answer.
  """

  def setUp(self):
    self.config = _make_config()

  def _check(self, completions, answer):
    return utils_rl.check_numbers(
        prompts=None,
        completions=completions,
        answer=answer,
        tmvp_config=self.config,
        question=["test question"] * len(completions),
    )

  # ---------------------------------------------------------------
  # Scenario 1: regex extraction succeeds / fails
  # ---------------------------------------------------------------

  def test_extraction_succeeds_full_format(self):
    """Full <reasoning>…</reasoning><answer>…</answer> format allows extraction."""
    scores = self._check(
        completions=["<reasoning>40 + 2 = 42</reasoning><answer>42</answer>"],
        answer=['["42"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  def test_extraction_fails_no_tags(self):
    """Plain-text completion without any tags yields score 0 (cannot extract)."""
    scores = self._check(
        completions=["The answer is 42."],
        answer=['["42"]'],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_format)

  def test_extraction_fails_answer_tags_only(self):
    """<answer> tag alone (no <reasoning> block) is matched by the regex as a fallback, score 1.5."""
    scores = self._check(
        completions=["<answer>42</answer>"],
        answer=['["42"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  def test_extraction_fails_reasoning_tags_only(self):
    """<reasoning> block with no <answer> tag cannot be extracted, score 0."""
    scores = self._check(
        completions=["<reasoning>The answer is 42.</reasoning>"],
        answer=['["42"]'],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_format)

  def test_extraction_batch_mixed(self):
    """Batch with one extractable and one non-extractable completion."""
    scores = self._check(
        completions=[
            "<reasoning>thinking</reasoning><answer>7</answer>",  # extractable
            "just 7",  # not extractable
        ],
        answer=['["7"]', '["7"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)
    self.assertEqual(scores[1], self.config.penalty_incorrect_format)

  def test_extraction_for_mcq(self):
    """Batch with two multiple-choice questions and one single-answer question."""
    scores = self._check(
        completions=[
            "<reasoning>thinking</reasoning><answer>7</answer>",
            "<reasoning>thinking</reasoning><answer>A</answer>",
            "<reasoning>thinking</reasoning><answer>A</answer>",
        ],
        answer=['["7", "B"]', '["7", "A"]', '["7", "7"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)
    self.assertEqual(scores[1], self.config.reward_exact_answer)
    self.assertEqual(scores[2], self.config.penalty_incorrect_answer)  # extracted "A" does not match "7"

  # ---------------------------------------------------------------
  # Scenario 2: extraction succeeds, value matches/mismatches the answer
  # ---------------------------------------------------------------

  def test_extracted_matches_integer_answer(self):
    """Extracted integer equal to reference answer earns 1.5."""
    scores = self._check(
        completions=["<reasoning>simple</reasoning><answer>100</answer>"],
        answer=['["100"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  def test_extracted_does_not_match_answer(self):
    """Extracted number that differs from the reference answer earns 0.0."""
    scores = self._check(
        completions=["<reasoning>wrong path</reasoning><answer>99</answer>"],
        answer=['["42"]'],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_answer)

  def test_extracted_matches_comma_formatted_number(self):
    """Comma-formatted guess (e.g. '1,000') normalizes to match integer answer '1000'."""
    scores = self._check(
        completions=["<reasoning>cost calculation</reasoning><answer>1,000</answer>"],
        answer=['["1000"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  def test_extracted_matches_with_currency_prefix(self):
    """Leading '$' in extracted answer is normalized away before comparison."""
    scores = self._check(
        completions=["<reasoning>price is $16</reasoning><answer>$16</answer>"],
        answer=['["16"]'],
    )
    self.assertEqual(scores[0], self.config.reward_exact_answer)

  def test_extracted_non_numeric_no_match(self):
    """Non-numeric extraction that cannot be float-converted and does not math-verify returns 0."""
    scores = self._check(
        completions=["<reasoning>thinking</reasoning><answer>blue</answer>"],
        answer=['["red"]'],
    )
    self.assertEqual(scores[0], self.config.penalty_incorrect_format)

  def test_duplicate_guesses_are_graded_once(self):
    """Completions with the same answer send one job to the math_verify pool and all get its score."""
    submitted = []

    class _InProcessPool:

      def apply_async(self, fn, args):
        submitted.append(args)
        value = fn(*args)
        return SimpleNamespace(ready=lambda: True, get=lambda timeout=None: value)

    completions = ["<reasoning>r</reasoning><answer>\\frac{1}{2}</answer>"] * 3 + [
        "<reasoning>r</reasoning><answer>0.25</answer>"
    ]
    with mock.patch.object(math_verify_pool, "_get_pool", return_value=_InProcessPool()):
      scores = self._check(completions=completions, answer=['["0.5"]'] * 4)
    self.assertEqual(len(submitted), 2)
    self.assertEqual(scores, [self.config.reward_exact_answer] * 3 + [self.config.penalty_incorrect_format])

  def test_debug_log_is_written_off_thread_and_failures_are_logged(self):
    with tempfile.TemporaryDirectory() as tmp:
      utils_rl._write_debug_log_async(tmp, "run", "log text")  # pylint: disable=protected-access
      files = list((pathlib.Path(tmp) / "run" / "debug_rl_logs").glob("check_numbers_*.txt"))
      self.assertEqual([f.read_text(encoding="utf-8") for f in files], ["log text"])

      not_a_dir = os.path.join(tmp, "file")
      pathlib.Path(not_a_dir).write_text("", encoding="utf-8")
      with mock.patch.object(utils_rl.max_logging, "log") as log:
        utils_rl._write_debug_log_async(not_a_dir, "run", "log text")  # pylint: disable=protected-access
      log.assert_called_once()


class TestRewardFastPathsKeepSemantics(unittest.TestCase):
  """The faster reward helpers must return what the code they replaced returned."""

  def test_match_format_exactly_matches_the_format_regex(self):
    config = _make_config()
    completions = [
        "",
        "<reasoning>a</reasoning><answer>4</answer>",
        "<reasoning>a</reasoning><answer></answer>",
        "<answer>4</answer><reasoning>a</reasoning>",
        "<reasoning>a</reasoning>x<answer>4",
        "</reasoning><answer>4</answer>",
        "<reasoning></reasoning>\n<answer>4</answer><answer>",
        "<reasoning>" + "x" * 500 + "</reasoning>" + "<answer>" * 20,
    ]
    for start_in_prompt in (False, True):
      config.reasoning_start_token_in_prompt = start_in_prompt
      start = "" if start_in_prompt else re.escape(config.reasoning_start_token)
      reference = re.compile(
          rf"{start}.*?{re.escape(config.reasoning_end_token)}.*?"
          rf"{re.escape(config.solution_start_token)}(.+?){re.escape(config.solution_end_token)}",
          flags=re.MULTILINE | re.DOTALL,
      )
      expected = [bool(reference.search(c)) for c in completions]
      with self.subTest(start_in_prompt=start_in_prompt):
        scores = utils_rl.match_format_exactly(None, completions, config)
        self.assertEqual(scores, [config.reward_exact_format_match if e else 0 for e in expected])
        match_format = utils_rl.get_match_format_regex(config)  # Also used by evaluate_rl.
        self.assertEqual([bool(match_format.search(c)) for c in completions], expected)

  def test_units_are_removed_one_unit_at_a_time(self):
    # Removing a unit can join its neighbors into another unit, so UNITS order
    # matters. Expected values come from the original per-unit loop.
    cases = {
        "mmeterm": "",
        "12cmonth": "12c",
        "inchesecond": "e",
        "degreemetersecond": "",
        "5 square meters": "5",
        "10 cm^2": "10",
        "7 3/4 inches": "7+3/4",
    }
    for text, expected in cases.items():
      self.assertEqual(utils_rl.normalize_final_answer(text), expected, text)

  def test_fix_latex_escaping(self):
    cases = {
        "textbf{x} + text": "\\textbf{x} + \\text",
        "x to infinity": "x \\to infinity",
        "leftarrow": "\\leftarrow",
        "pi r^2": "\\pi r^2",
        "sqrt2 \\pi": "sqrt2 \\pi",
        "tan theta": "\\tan \\theta",
        "hat{x}": "\\hat{x}",
        "photo": "photo",
        "\tan x": "\\tan x",  # "\t" became a tab.
        "\x08oxed{2}": "\\boxed{2}",  # "\b" became a backspace.
    }
    for text, expected in cases.items():
      self.assertEqual(utils_rl.fix_latex_escaping(text), expected, repr(text))

  def test_extract_answer_edge_cases(self):
    config = _make_config()
    cases = {
        "": utils_rl.FALLBACK_ANSWER,
        "no answer here": utils_rl.FALLBACK_ANSWER,
        "<reasoning>r</reasoning><answer> 42 </answer>": "42",
        "<answer>1</answer> then <answer>\\boxed{2}</answer>": "2",
        "{{ \\boxed{5}": "5",  # Unmatched braces before the box.
        "}} x \\boxed{6} {": "6",
        "\\boxed{\\frac{1}{2}}": "\\frac{1}{2}",
        "\\boxed{7": "7",  # Unclosed box: permissive fallback regex.
        "<answer>\\boxed{3}": "3",  # No closing tag: search the whole response.
        "\\boxed{1} and \\boxed{2}": "2",
        "<answer>x</answer> \\boxed{9}": "x",  # The last answer block wins over a box outside it.
    }
    for response, expected in cases.items():
      self.assertEqual(utils_rl.extract_answer(response, config), expected, response)


class TestExtractHashAnswer(unittest.TestCase):
  """Tests for utils_rl.extract_hash_answer."""

  def test_with_hash(self):
    """Test extraction when #### is present."""
    self.assertEqual(utils_rl.extract_hash_answer("The answer is #### 42"), "42")
    self.assertEqual(utils_rl.extract_hash_answer("Some reasoning ####   123.45  "), "123.45")
    self.assertEqual(utils_rl.extract_hash_answer("####"), "")

  def test_without_hash(self):
    """Test extraction when #### is not present."""
    self.assertIsNone(utils_rl.extract_hash_answer("The answer is 42"))
    self.assertIsNone(utils_rl.extract_hash_answer(""))


class TestGetOptimizer(unittest.TestCase):
  """Tests for utils_rl.get_optimizer."""

  def _make_optimizer_config(self, gradient_clipping_threshold=0.0):
    return SimpleNamespace(
        learning_rate=1e-4,
        warmup_steps_fraction=0.1,
        gradient_clipping_threshold=gradient_clipping_threshold,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_weight_decay=0.01,
        train_steps=100,
    )

  def test_returns_optimizer_without_clipping(self):
    """get_optimizer returns an optax optimizer when gradient clipping is disabled."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    config = self._make_optimizer_config(gradient_clipping_threshold=0.0)
    opt = utils_rl.get_optimizer(config)
    # Should be usable: init on a simple param tree
    params = {"w": jnp.ones(3)}
    state = opt.init(params)
    self.assertIn("learning_rate", state.hyperparams)

  def test_returns_optimizer_with_clipping(self):
    """get_optimizer includes gradient clipping when threshold > 0."""
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

    config = self._make_optimizer_config(gradient_clipping_threshold=1.0)
    opt = utils_rl.get_optimizer(config)
    params = {"w": jnp.ones(3)}
    state = opt.init(params)
    self.assertIn("learning_rate", state.hyperparams)


class TestFormatMaxTextMessages(unittest.TestCase):
  """Tests for utils_rl.format_maxtext_messages."""

  def setUp(self):
    self.config = _make_config()
    self.template_config = {
        "SYSTEM_PROMPT": "Reason between {reasoning_start_token} and {reasoning_end_token}. "
        + "Solution between {solution_start_token} and {solution_end_token}.",
        "TEMPLATE": "system: {system_prompt}\nquestion: {question}",
    }

  def test_format_with_template(self):
    """Test formatting when a template is provided."""
    messages = ["What is 2+2?"]
    formatted = utils_rl.format_maxtext_messages(messages, self.template_config, self.config)
    self.assertEqual(len(formatted), 1)
    self.assertEqual(formatted[0]["role"], "user")
    expected_content = (
        "system: Reason between <reasoning> and </reasoning>. "
        "Solution between <answer> and </answer>.\n"
        "question: What is 2+2?"
    )
    self.assertEqual(formatted[0]["content"], expected_content)

  def test_format_without_template(self):
    """Test formatting when template_config is None (the fix)."""
    messages = ["What is 2+2?"]
    self.assertRaises(ValueError, lambda: utils_rl.format_maxtext_messages(messages, None, self.config))


if __name__ == "__main__":
  unittest.main()
