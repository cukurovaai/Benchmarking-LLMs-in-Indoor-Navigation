from pathlib import Path
import json
from typing import Dict, List, Tuple, Union

from omegaconf import DictConfig
import numpy as np
import hydra
import habitat_sim

from vlmaps.task.habitat_task import HabitatTask
from vlmaps.utils.habitat_utils import agent_state2tf, get_position_floor_objects
from vlmaps.utils.navigation_utils import get_dist_to_bbox_2d
from vlmaps.utils.habitat_utils import display_sample


class HabitatSpatialGoalNavigationTask(HabitatTask):
    def load_task(self):
        assert hasattr(self, "vlmaps_dataloader"), "Please call setup_scene() first"

        task_path = Path(self.vlmaps_dataloader.data_dir) / "llm_prompts.json"
        with open(task_path, "r") as f:
            self.task_dict = json.load(f)

    def setup_task(self, task_id: int):
        json_task_id = self.task_dict[task_id]["task_id"]
        assert json_task_id == task_id, "Task ID mismatch"
        self.task_id = task_id
        self.init_hab_tf = np.array(self.task_dict[task_id]["tf_habitat"], dtype=np.float32).reshape((4, 4))
        self.map_grid_size = self.task_dict[task_id]["map_grid_size"]
        self.map_cell_size = self.task_dict[task_id]["map_cell_size"]
        self.scene = self.task_dict[task_id]["scene"]
        self.instruction = self.task_dict[task_id]["instruction"]
        self.actions = []

    def test_step(
        self, sim: habitat_sim.Simulator, action: str, task: str, task_id: str, agent_map_position: np.array = None, vis: bool = False
    ):
        self.actions.append(action)
        if action != "stop":
            sim.step(action)
            obs = sim.get_sensor_observations(0)
            display_sample({}, obs["color_sensor"], task, task_id, waitkey=True)
        if agent_map_position is None:
            agent = sim.get_agent(0)
            agent_state = agent.get_state()

            habitat_2_cam_pose = np.eye(4)
            habitat_2_cam_pose[1, 1] = -1
            habitat_2_cam_pose[2, 2] = -1
            tf_hab = agent_state2tf(agent_state)
            self.vlmaps_dataloader.from_habitat_tf(tf_hab)
            agent_map_position = self.vlmaps_dataloader.to_full_map_pose()[:2]
            # tf_cam = tf_hab @ self.tf_ro_cam
            # pose = np.linalg.inv(self.map_init_cam_pose) @ tf_cam
            # x, y = pos2grid_id(self.gs, self.cs, pose[0, 3], pose[2, 3])
            # agent_map_position = [x, y]

    def save_single_task_metric(
        self,
        save_path: Union[Path, str],
        forward_dist: float = 0.05,
        turn_angle: float = 1,
    ):
        results_dict = {}
        results_dict["task_id"] = self.task_id
        results_dict["scene"] = self.scene
        results_dict["num_subgoals"] = self.n_subgoals_in_task
        # results_dict["num_subgoal_success"] = self.n_success_subgoals
        results_dict["subgoal_success_rate"] = self.subgoal_success_rate
        results_dict["finished_subgoal_ids"] = self.finished_subgoals
        results_dict["instruction"] = self.instruction
        results_dict["forward_dict"] = forward_dist
        results_dict["turn_angle"] = turn_angle
        results_dict["init_tf_hab"] = self.init_hab_tf.tolist()
        results_dict["actions"] = self.actions
        with open(save_path, "w") as f:
            json.dump(results_dict, f, indent=4)


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="test_config.yaml",
)
def main(config: DictConfig) -> None:
    from vlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat

    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    dataloader = VLMapsDataloaderHabitat(data_dirs[0], config.map_config)
    task = HabitatSpatialGoalNavigationTask(config)
    task.setup_scene(dataloader)
    task.load_task()
    task.setup_task(0)
    # print(task.goals)


if __name__ == "__main__":
    main()
