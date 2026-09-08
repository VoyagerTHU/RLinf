"""Write a deterministic file-transport request for StarVLA bridge checks."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from deployment.model_server.tools import msgpack_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = np.arange(224, dtype=np.uint8)[:, None]
    cols = np.arange(224, dtype=np.uint8)[None, :]
    image = np.stack(
        [
            np.broadcast_to(rows, (224, 224)),
            np.broadcast_to(cols, (224, 224)),
            (rows.astype(np.uint16) + cols.astype(np.uint16)).astype(np.uint8),
        ],
        axis=-1,
    )
    query = {
        "examples": [
            {
                "image": [image],
                "lang": "pick the cup and place it in the drawer then close the drawer",
                "state": np.linspace(-1, 1, 58, dtype=np.float32),
            }
        ],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
    }

    request_id = f"equivalence-{os.getpid()}"
    request = args.bridge_dir / f"request.{request_id}.msgpack"
    response = args.bridge_dir / f"response.{request_id}.msgpack"
    packer = msgpack_numpy.Packer()
    temporary = request.with_suffix(request.suffix + ".tmp")
    temporary.write_bytes(packer.pack(query))
    os.replace(temporary, request)
    deadline = time.time() + 300
    while not response.exists():
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {response}")
        time.sleep(0.01)
    payload = msgpack_numpy.unpackb(response.read_bytes())
    response.unlink()
    if not payload.get("ok", False):
        raise RuntimeError(f"Inference request failed: {payload.get('error', payload)}")
    data = payload["data"]
    key = "normalized_actions" if "normalized_actions" in data else "env_actions"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(data[key]))
    print(f"saved {key} {np.asarray(data[key]).shape} to {args.output}")


if __name__ == "__main__":
    main()
