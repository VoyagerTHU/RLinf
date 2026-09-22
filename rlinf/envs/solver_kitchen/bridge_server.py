# Copyright 2025 The RLinf Authors.
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

"""Standalone solver process driven by :class:`SubprocessBackend`.

Run with the *solver* interpreter (Python 3.11 + Newton/Warp), never with the
RLinf training interpreter::

    python bridge_server.py --address /tmp/solver.sock --authkey <hex>

The protocol is a simple request/response loop over
:mod:`multiprocessing.connection` (pickle). Requests are ``(command, payload)``
tuples and every response is ``("ok", result)`` or ``("error", message)``.
This file deliberately avoids importing ``rlinf`` so it can execute in an
interpreter that has no RLinf dependencies installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


def _load_core_module(core_path: Path | None):
    """Import ``solver_core`` by file path so no ``rlinf`` package is needed."""
    if core_path is None:
        core_path = Path(__file__).resolve().with_name("solver_core.py")
    spec = importlib.util.spec_from_file_location("solver_kitchen_core", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solver core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BridgeServer:
    """Dispatch bridge commands onto a core object."""

    def __init__(self, core_factory) -> None:
        self._core_factory = core_factory
        self._core = None

    def handle(self, command: str, payload: Any) -> Any:
        if command == "ping":
            return {"protocol": PROTOCOL_VERSION, "pid": os.getpid()}
        if command == "init":
            if self._core is not None:
                raise RuntimeError("solver core already initialized")
            self._core = self._core_factory(payload)
            return self._core.metadata()
        if command == "close":
            if self._core is not None:
                self._core.close()
                self._core = None
            return None
        if self._core is None:
            raise RuntimeError(f"command {command!r} requires init first")
        if command == "metadata":
            return self._core.metadata()
        if command == "reset":
            env_ids = payload.get("env_ids")
            seeds = payload.get("seeds")
            return self._core.reset(
                None if env_ids is None else np.asarray(env_ids),
                None if seeds is None else np.asarray(seeds),
            )
        if command == "step":
            return self._core.step(np.asarray(payload["actions"]))
        if command == "render":
            return self._core.render()
        if command == "states":
            return self._core.states()
        raise ValueError(f"unknown bridge command {command!r}")

    def serve(self, address: str, authkey: bytes) -> None:
        with Listener(address, family="AF_UNIX", authkey=authkey) as listener:
            with listener.accept() as conn:
                while True:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break
                    command, payload = request
                    try:
                        result = self.handle(command, payload)
                        conn.send(("ok", result))
                    except Exception:  # noqa: BLE001 - forward everything
                        conn.send(("error", traceback.format_exc()))
                        if command == "init":
                            # A failed init leaves nothing to serve.
                            break
                    if command == "close":
                        break
        if self._core is not None:
            self._core.close()


def _default_core_factory(core_module):
    def factory(config_payload: dict[str, Any]):
        config = core_module.SolverKitchenCoreConfig.from_dict(config_payload)
        return core_module.SolverKitchenCore(config)

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="Unix socket path")
    parser.add_argument("--authkey", required=True, help="hex-encoded auth key")
    parser.add_argument(
        "--core-path",
        default=None,
        help="Path to solver_core.py (default: next to this file)",
    )
    parser.add_argument(
        "--fake-core",
        default=None,
        help="module:attr of a fake core factory used by tests",
    )
    args = parser.parse_args(argv)

    if args.fake_core:
        module_name, attr = args.fake_core.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr)
    else:
        core_module = _load_core_module(
            None if args.core_path is None else Path(args.core_path)
        )
        factory = _default_core_factory(core_module)

    server = BridgeServer(factory)
    server.serve(args.address, bytes.fromhex(args.authkey))
    return 0


if __name__ == "__main__":
    sys.exit(main())
