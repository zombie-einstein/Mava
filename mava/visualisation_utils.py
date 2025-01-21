# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from flax.typing import FrozenDict
from jax.random import PRNGKey

from mava.evaluator import ActorState, EvalActFn
from mava.types import MarlEnv


def rollout_mava_system(
    env: MarlEnv, params: FrozenDict, actor_state: ActorState, key: PRNGKey, act_fn: EvalActFn
) -> list:
    env_step = jax.jit(env.step)
    jit_act = jax.jit(act_fn)

    key, env_key = jax.random.split(key)
    state, ts = env.reset(env_key)
    actor_state = jax.tree.map(lambda x: x[:, 0], actor_state)

    states = []

    # Only loop once
    while not ts.last():
        # Eval env is wrapped in the record metrics wrapper. We just store the true env state
        # for jumanji to be able to render.
        states.append(state.env_state)
        ts = jax.tree_map(lambda x: x[jnp.newaxis, ...], ts)
        key, act_key = jax.random.split(key, 2)
        action, actor_state = jit_act(params, ts, act_key, actor_state)
        # note: dangerous squeeze, but we don't want a batch or time dim here
        state, ts = env_step(state, action.squeeze())

    return states
