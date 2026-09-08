"""Tests for the RoboCasa-GR1 drawer task's physical milestone signals."""

import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import robocasa.environments.tabletop.tabletop_drawer_pnp as drawer_pnp
except ImportError:
    drawer_pnp = None


def make_task():
    """Construct the minimum task state needed by the signal methods."""
    task = drawer_pnp.TabletopDrawerPnPClose.__new__(drawer_pnp.TabletopDrawerPnPClose)
    task.objects = {"obj": SimpleNamespace(name="target", root_body="target")}
    task.drawer = SimpleNamespace(name="drawer")
    task.robots = [SimpleNamespace(gripper={"left": object()})]
    task._check_grasp = lambda **_kwargs: False
    return task


@unittest.skipIf(drawer_pnp is None, "RoboCasa-GR1 task package is not installed")
class DrawerSubtaskSignalTest(unittest.TestCase):
    """Verify that reward milestones match the task's success geometry."""

    def test_obj_in_drawer_uses_containment_check(self):
        """Drawer contact alone must not satisfy the in-drawer milestone."""
        task = make_task()

        with (
            mock.patch.object(
                drawer_pnp, "obj_inside_of", autospec=True, return_value=False
            ) as containment_check,
            mock.patch.object(
                drawer_pnp.OU,
                "check_obj_fixture_contact",
                autospec=True,
                return_value=True,
            ),
        ):
            signals = task.get_subtask_term_signals()

        self.assertEqual(signals["obj_in_drawer"], 0)
        containment_check.assert_called_once_with(
            env=task,
            obj_name="target",
            fixture_id=task.drawer,
            partial_check=True,
        )

    def test_success_and_milestone_share_containment_semantics(self):
        """Success should add only the drawer-door condition to containment."""
        task = make_task()
        task.behavior = "close"
        task.drawer.get_door_state = lambda **_kwargs: {"door": 0.0}

        with mock.patch.object(
            drawer_pnp, "obj_inside_of", autospec=True, return_value=True
        ):
            self.assertEqual(task.get_subtask_term_signals()["obj_in_drawer"], 1)
            self.assertTrue(task._check_success())


if __name__ == "__main__":
    unittest.main()
