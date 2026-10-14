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

"""Tests that Attention.forward_serve_vllm forwards tpu_inference's decode-only flag."""

import sys
import types
import unittest
from unittest import mock

import jax.numpy as jnp

from maxtext.common.common_types import AttentionType
from maxtext.layers.attentions import Attention

_RPA_MODULE = "tpu_inference.layers.common.attention_interface"


def _metadata(**extra_fields):
  return types.SimpleNamespace(
      seq_lens=jnp.ones((2,), jnp.int32),
      block_tables=jnp.zeros((2,), jnp.int32),
      query_start_loc=jnp.arange(3, dtype=jnp.int32),
      request_distribution=jnp.array([2, 2, 2], jnp.int32),
      **extra_fields,
  )


class ForwardServeVllmDecodeOnlyTest(unittest.TestCase):
  """forward_serve_vllm with a stub RPA op, so tpu_inference is not needed."""

  def _rpa_kwargs(self, metadata):
    """Runs forward_serve_vllm on 2 decode tokens and returns the RPA op's kwargs."""
    rpa_ops = mock.Mock(return_value=(jnp.zeros((2, 4, 128)), ["new_kv_cache"]))
    fake_module = types.ModuleType(_RPA_MODULE)
    fake_module.sharded_ragged_paged_attention = rpa_ops
    layer = types.SimpleNamespace(
        config=types.SimpleNamespace(attention="vllm_rpa", sliding_window_size=0),
        dtype=jnp.float32,
        attention_type=AttentionType.GLOBAL,
        share_kv_layer=False,
        sinks=None,
        mesh=None,
    )
    query = jnp.ones((2, 1, 4, 128))
    key_value = jnp.ones((2, 1, 2, 128))
    with mock.patch.dict(sys.modules, {_RPA_MODULE: fake_module}):
      _, kv_cache = Attention.forward_serve_vllm(
          layer, query, key_value, key_value, rpa_kv_cache=["kv_cache"], rpa_metadata=metadata
      )
    self.assertEqual(kv_cache, ["new_kv_cache"])
    rpa_ops.assert_called_once()
    return rpa_ops.call_args.kwargs

  def test_decode_only_metadata_is_forwarded(self):
    # The fused decode loop marks its metadata decode-only, so the RPA op
    # skips its empty prefill and mixed launches.
    self.assertIs(self._rpa_kwargs(_metadata(decode_only=True))["decode_only"], True)

  def test_other_metadata_keeps_the_stock_call(self):
    self.assertNotIn("decode_only", self._rpa_kwargs(_metadata(decode_only=False)))
    # Older tpu_inference versions have no decode_only field (and no kwarg).
    self.assertNotIn("decode_only", self._rpa_kwargs(_metadata()))


if __name__ == "__main__":
  unittest.main()
