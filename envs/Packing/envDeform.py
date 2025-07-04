from typing import Optional

from .deformContainer import DeformContainer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .cutCreator import CuttingBoxCreator
# from .mdCreator import MDlayerBoxCreator
from .binCreator import RandomDeformBoxCreator, LoadBoxCreator, BoxCreator

from render import VTKRender


class DeformPackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None, 
        data_name=None, 
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type="terminal",
        action_scheme="heightmap",
        k_placement=100,
        is_render=False,
        is_hold_on=False,
        use_test_indices= False,
        save_img_path=None,
        **kwags
    ) -> None:
        self.use_test_data = use_test_indices
        self.test_index = 0
        self.save_img_path = save_img_path
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        # packing state
        self.container = DeformContainer(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        self.k_placement = k_placement
        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)  # (x1, y1, z1, x2, y2, H)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)  # (x, y, z)

        # Generator for train/test data
        if not load_test_data:
            # assert item_set is not None
            if data_type == "random":
                print(f"using items generated randomly")
                self.box_creator = RandomDeformBoxCreator(item_set)  
            if data_type == "cut":
                print(f"using items generated through cutting method")
                low = list(item_set[0])
                up = list(item_set[-1])
                low.extend(up)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            assert isinstance(self.box_creator, BoxCreator)
        if load_test_data:
            print(f"use box dataset: {data_name}")
            self.box_creator = LoadBoxCreator(data_name)

        self.test = load_test_data

        # for rendering
        self.is_render = is_render
        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        
        self._set_space()

    def _set_space(self) -> None:
        # nisara: Add dimensions for stressmap and fragility map if they need to be added to the policy - DO NOT NEED TO BE ADDED
        obs_len = self.area + 12  # the state of bin -> heightmap + the dimension of box (l, w, h)
        obs_len += self.k_placement * 6 # number of allowed EMS placements * 6 (dimension of each EMS placement representation) - this is not the mask, but the actual EMS coordinates
        self.action_space = spaces.Discrete(self.k_placement)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=max(self.bin_size), shape=(obs_len, )), # highest value the obs can take is the size of the bin
                "mask": spaces.Discrete(self.k_placement)
            }
        )

    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
                self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])

    # box mask (W x L x 3)
    def get_box_plain(self):
        coming_box = self.next_box
        x_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[0]
        y_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[1]
        z_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[2]
        return x_plain, y_plain, z_plain

    @property
    def cur_observation(self):
        """
            get current observation and action mask
        """
        hmap = self.container.heightmap
        size = list(self.next_box)
        placements, mask = self.get_possible_position(size)
        self.candidates = np.zeros_like(self.candidates)
        if len(placements) != 0:
            # print("candidates:")
            # for c in placements:
            #     print(c)
            self.candidates[0:len(placements)] = placements

        size.extend([size[1], size[0], size[2], size[3], size[4], size[5]])
        # size becomes (l, w, h, m, k, f, w, l, h, m, k , f) here 
        obs = np.concatenate((hmap.reshape(-1), np.array(size).reshape(-1), self.candidates.reshape(-1)))
        mask = mask.reshape(-1)
        return {
            "obs": obs, 
            "mask": mask
        }

    @property
    def next_box(self) -> list:
        return self.box_creator.preview(1)[0]

    def get_possible_position(self, next_box):
        """
            get possible actions for next box
        Args:
            scheme: the scheme how to generate candidates

        Returns:
            candidate action mask, i.e., the position where the current item should be placed
        """
        box_properties = next_box[3], next_box[4], next_box[5]  # mass, spring constant, fragility index
        if self.action_scheme == "heightmap":
            candidates = self.container.candidate_from_heightmap(next_box, box_properties, self.k_placement)
        elif self.action_scheme == "EP":
            candidates, mask = self.container.candidate_from_EP(next_box, box_properties, self.k_placement)
        elif self.action_scheme == "EMS":
            candidates, mask = self.container.candidate_from_EMS(next_box, box_properties, self.k_placement)
        elif self.action_scheme == "FC": # full coordinate space
            candidates, mask = self.container.candidate_from_FC(next_box, box_properties)
        else:
            raise NotImplementedError("action scheme not implemented")

        return candidates, mask 

    def idx2pos(self, idx):
        if idx >= self.k_placement - 1:
            idx = idx - self.k_placement
            rot = 1
        else:
            rot = 0

        pos = self.candidates[idx][:3]

        if rot == 1:
            dim = [self.next_box[1], self.next_box[0], self.next_box[2]]
        else:
            dim = list(self.next_box)
        self.render_box = [dim, pos]

        return pos, rot, dim

    # READ: Post package deformation is added to place_box function
    def step(self, action):
        """

        :param action: action index
        :return: cur_observation
                 reward
                 done, Whether to end boxing (i.e., the current box cannot fit in the bin)
                 info
        """
        # print(self.next_box)
        pos, rot, size = self.idx2pos(action)
 
        properties_next_box = self.next_box[3], self.next_box[4], self.next_box[5]  # mass, spring constant, fragility index
        succeeded = self.container.place_box(self.next_box, pos, rot, properties_next_box)
        
        if not succeeded:
            if self.reward_type == "terminal":  # Terminal reward
                # reward = self.container.get_volume_ratio()
                ## NOTE: If the reward is the volume_ratio, it might learn not to allow deformable objects in the fear that it will reduce the volume in later iterations
                reward = len(self.container.boxes)
            else:  # Step-wise/Immediate reward
                reward = 0.0
            done = True
            # print("Episode finished here and the reward is: ", reward)
            self.test_index += 1
            if self.is_render:
                # self.renderer.hold_on()
                self.renderer.save_img(self.save_img_path, self.test_index)
                self.renderer.combine_frames_to_video(self.save_img_path, self.test_index)
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
            return self.cur_observation, reward, done, False, info

        box_ratio = self.get_box_ratio()

        self.box_creator.drop_box()  # remove current box from the list
        self.box_creator.generate_box_size(self.use_test_data, self.test_index)  # add a new box to the list

        if self.reward_type == "terminal":
            ## nisara: CHANGED: reward is the volume ratio of the box to the bin
            # reward = 0.01
            # reward = self.container.get_volume_ratio()
            reward = len(self.container.boxes)
        else:
            reward = box_ratio
        done = False

        info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}

        return self.cur_observation, reward, done, False, info

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.box_creator.reset()
        self.container = DeformContainer(*self.bin_size)
        self.box_creator.generate_box_size(self.use_test_data, self.test_index)
        self.candidates = np.zeros_like(self.candidates)
        return self.cur_observation, {}
    
    def seed(self, s=None):
        np.random.seed(s)

    def render(self):
        self.renderer.render_deform_bin(self.container.box_id_map_3d)
        # self.renderer.add_item(self.render_box[0], self.render_box[1])
        # self.renderer.save_img()
        self.renderer.save_video_frame(self.save_img_path)
