import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import hydra

from vlmaps.task.habitat_spatial_goal_nav_task import HabitatSpatialGoalNavigationTask
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_spatial_goal_instruction
from vlmaps.utils.matterport3d_categories import mp3dcat

from vlmaps.utils.habitat_utils import *
from typing import Optional, Sequence, Tuple
import imageio.v2 as imageio
import scipy.ndimage
import quaternion


MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_SHORTEST_PATH_COLOR = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[10:] = cv2.applyColorMap(np.arange(246, dtype=np.uint8), cv2.COLORMAP_JET).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_COLOR] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Green


def quaternion_rotate_vector(
    quat: quaternion.quaternion, v: np.ndarray
) -> np.ndarray:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = quaternion.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background

def quaternion_to_yaw(rotation) -> float:
    """
    Converts a quaternion to the yaw angle (rotation around the Y-axis).
    """
    """
    # Convert quaternion to a scipy Rotation object
    r = R.from_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    # Extract yaw (rotation around Y-axis) in radians
    yaw = r.as_euler('xyz', degrees=False)[1]
    """

    heading_vector = quaternion_rotate_vector(
        rotation.inverse(), np.array([0, 0, -1])
    )

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    yaw = phi - np.pi / 2

    return yaw - 1.5708

def draw_agent(
    image: np.ndarray,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 20,
) -> np.ndarray:
    r"""Return an image with the agent image composited onto it.
    Args:
        image: the image onto which to put the agent.
        agent_center_coord: the image coordinates where to paste the agent.
        agent_rotation: the agent's current rotation in radians.
        agent_radius_px: 1/2 number of pixels the agent will be resized to.
    Returns:
        The modified background image. This operation is in place.
    """
    agent_sprite = imageio.imread("./agent_sprite.png")
    agent_sprite = np.ascontiguousarray(np.flipud(agent_sprite))

    # Rotate before resize to keep good resolution.
    rotated_agent = scipy.ndimage.rotate(
        agent_sprite, agent_rotation * 180 / np.pi
    )
    # Rescale because rotation may result in larger image than original, but
    # the agent sprite size should stay the same.
    initial_agent_size = agent_sprite.shape[0]
    new_size = rotated_agent.shape[0]
    agent_size_px = max(
        1, int(agent_radius_px * 2 * new_size / initial_agent_size)
    )
    resized_agent = cv2.resize(
        rotated_agent,
        (agent_size_px, agent_size_px),
        interpolation=cv2.INTER_LINEAR,
    )
    
    ret_img = paste_overlapping_image(image, resized_agent, agent_center_coord)
    return ret_img

def draw_path(
    top_down_map: np.ndarray,
    path_points: Sequence[Tuple],
    color: int = 10,
    thickness: int = 6,
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
    Args:
        top_down_map: A colored version of the map.
        color: color code of the path, from TOP_DOWN_MAP_COLORS.
        path_points: list of points that specify the path to be drawn
        thickness: thickness of the path.
    """
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            prev_pt[::-1],
            next_pt[::-1],
            (255, 0, 0),
            thickness,
        )  # type: ignore
    
    return top_down_map

def to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: Tuple[int, int],
    sim: Optional["HabitatSim"] = None,
    pathfinder=None,
) -> Tuple[int, int]:
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()

    grid_size = (
        abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
        abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - lower_bound[2]) / grid_size[0])
    grid_y = int((realworld_y - lower_bound[0]) / grid_size[1])
    return grid_x, grid_y

def _outline_border(top_down_map):
    left_right_block_nav = (top_down_map[:, :-1] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )
    left_right_nav_block = (top_down_map[:, 1:] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )

    up_down_block_nav = (top_down_map[:-1] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )
    up_down_nav_block = (top_down_map[1:] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )

    top_down_map[:, :-1][left_right_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[:, 1:][left_right_nav_block] = MAP_BORDER_INDICATOR

    top_down_map[:-1][up_down_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[1:][up_down_nav_block] = MAP_BORDER_INDICATOR

def calculate_meters_per_pixel(
    map_resolution: int, sim: Optional["HabitatSim"] = None, pathfinder=None
):
    r"""Calculate the meters_per_pixel for a given map resolution"""
    if sim is None and pathfinder is None:
        raise RuntimeError(
            "Must provide either a simulator or pathfinder instance"
        )

    if pathfinder is None:
        pathfinder = sim.pathfinder

    lower_bound, upper_bound = pathfinder.get_bounds()
    return min(
        abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
        for coord in [0, 2]
    )

def get_topdown_map(
    pathfinder,
    height: float,
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.

    :param pathfinder: A habitat-sim pathfinder instances to get the map from
    :param height: The height in the environment to make the topdown map
    :param map_resolution: Length of the longest side of the map.  Used to calculate :p:`meters_per_pixel`
    :param draw_border: Whether or not to draw a border
    :param meters_per_pixel: Overrides map_resolution an

    :return: Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """

    if meters_per_pixel is None:
        meters_per_pixel = calculate_meters_per_pixel(
            map_resolution, pathfinder=pathfinder
        )

    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel, height=height
    ).astype(np.uint8)

    # Draw border if necessary
    if draw_border:
        _outline_border(top_down_map)

    return np.ascontiguousarray(top_down_map)


def get_topdown_map_from_sim(
    sim: "HabitatSim",
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
    agent_id: int = 0,
) -> np.ndarray:
    r"""Wrapper around :py:`get_topdown_map` that retrieves that pathfinder and heigh from the current simulator

    :param sim: Simulator instance.
    :param agent_id: The agent ID
    """

    top_down_map = get_topdown_map(
        sim.pathfinder,
        sim.get_agent(agent_id).state.position[1],
        map_resolution,
        draw_border,
        meters_per_pixel,
    )
    _map = TOP_DOWN_MAP_COLORS[top_down_map]
    return _map

def save_tasks(tasks, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for task in tasks:
            file.write(task + '\n')
    print(f"Tasks are written to {file_path}")

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="llm_navigation_cfg",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = HabitatLanguageRobot(config)
    spatial_nav_task = HabitatSpatialGoalNavigationTask(config)
    spatial_nav_task.reset_metrics()
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id
   
    sim_setting =  {"color_sensor": True, "depth_sensor": None, "semantic_sensor": None}
    custom_output_path = "/vlmaps/custom_output"
    os.makedirs(custom_output_path, exist_ok=True)
    exp_count = len([name for name in os.listdir(custom_output_path) if os.path.isdir(os.path.join(custom_output_path, name))])
    exp_dir_name = f"collect_pose_{exp_count + 1}"
    exp_dir = os.path.join(custom_output_path, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    task_arr = ["robot.with_object_on_left('counter')", "robot.move_east('window')"]
    for scene_i, scene_id in enumerate(scene_ids):
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat.copy())
        spatial_nav_task.setup_scene(robot.vlmaps_dataloader)
        spatial_nav_task.load_task()

        test_dir = os.path.join(exp_dir, f"test_{scene_i}")
        os.makedirs(test_dir, exist_ok=True)
        trajectory_dir = os.path.join(test_dir, "trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)
            
        # map_image = cv2.imread(f"sem_map_{scene_i + 1}.png")
        map_image = get_topdown_map_from_sim(robot.sim)
        cv2.imwrite(os.path.join(test_dir, "map.png"), map_image)
        
        spatial_nav_task.setup_task(scene_i)
        print(f"tasks:\n {task_arr}")
        robot.empty_recorded_actions()
        robot.set_agent_state(spatial_nav_task.init_hab_tf)

        for line in task_arr:
            exec(line)

        agent_states = []
        recorded_actions_list = robot.get_recorded_actions()
        robot.set_agent_state(spatial_nav_task.init_hab_tf)
        for action_i, action in enumerate(recorded_actions_list):
            spatial_nav_task.test_step(robot.sim, action, vis=config.nav.vis, task="empty", task_id=scene_i)

            obs = robot.sim.get_sensor_observations(0)
            save_obs(test_dir, sim_setting, obs, action_i, None)
            agent_states.append(robot.sim.get_agent(0).get_state())
            
        real_states = []
        for count, state in enumerate(agent_states):
            x, y = to_grid(state.position[2], state.position[0], (map_image.shape[0], map_image.shape[1]), robot.sim, robot.sim.pathfinder)
            real_states.append((x, y))
            path_image = draw_path(map_image.copy(), real_states)
            print(f"saving trajectory output {count} into {trajectory_dir}")
            agent_image = draw_agent(path_image.copy(), real_states[-1], quaternion_to_yaw(state.rotation))
            cv2.imwrite(os.path.join(trajectory_dir, f"path_{count}.png"), agent_image)

        save_states(test_dir, agent_states)

if __name__ == "__main__":
    main()
