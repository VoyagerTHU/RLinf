import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rlinf.runners.embodied_runner import EmbodiedRunner


def test_seed_rollout_table_includes_optional_subtask_rates():
    with tempfile.TemporaryDirectory() as temporary_directory:
        runner = EmbodiedRunner.__new__(EmbodiedRunner)
        runner.cfg = SimpleNamespace(
            env=SimpleNamespace(
                train=SimpleNamespace(task_name="drawer-task"),
                eval=SimpleNamespace(task_name="drawer-task"),
            ),
            runner=SimpleNamespace(
                logger=SimpleNamespace(log_path=temporary_directory)
            ),
        )
        runner.metric_logger = SimpleNamespace(logger_backends=[])
        metrics = [
            {
                "sample_seed": [11, 11, 22, 22],
                "sample_group": [0, 0, 1, 1],
                "sample_trajectory": [0, 1, 0, 1],
                "success_once": [0.0, 1.0, 0.0, 0.0],
                "grasped_once": [1.0, 1.0, 1.0, 0.0],
                "obj_in_drawer_once": [0.0, 1.0, 1.0, 0.0],
                "grasp_first_step": [100, 90, 120, -1],
                "obj_in_drawer_first_step": [-1, 300, 350, -1],
                "success_first_step": [-1, 500, -1, -1],
                "episode_len": [720, 720, 720, 720],
                "return": [0.1, 1.0, 0.5, 0.0],
            }
        ]

        runner._log_seed_rollout_table(metrics, step=3, mode="train")

        table_path = (
            Path(temporary_directory) / "rollout_tables/train_seed_rollouts.jsonl"
        )
        rows = [json.loads(line) for line in table_path.read_text().splitlines()]
        assert len(rows) == 4
        assert rows[0]["trajectory_grasped"] == 1.0
        assert rows[0]["seed_grasp_rate"] == 1.0
        assert rows[0]["trajectory_obj_in_drawer"] == 0.0
        assert rows[0]["seed_obj_in_drawer_rate"] == 0.5
        assert rows[2]["seed_grasp_rate"] == 0.5
        assert rows[2]["seed_obj_in_drawer_rate"] == 0.5
        assert rows[1]["grasp_first_step"] == 90
        assert rows[1]["obj_in_drawer_first_step"] == 300
        assert rows[1]["success_first_step"] == 500
