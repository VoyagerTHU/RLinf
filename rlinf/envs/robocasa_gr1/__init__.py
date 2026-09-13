# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboCasa GR1 tabletop environment package.

The simulator-backed environment class is imported lazily so that modules
which only need the seed-pool helpers (config validation, unit tests) do not
pull in gymnasium and the simulator stack.
"""

from __future__ import annotations

__all__ = ["RoboCasaGR1Env"]


def __getattr__(name: str):
    if name == "RoboCasaGR1Env":
        from rlinf.envs.robocasa_gr1.robocasa_gr1_env import RoboCasaGR1Env

        return RoboCasaGR1Env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
