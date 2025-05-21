import copy
import itertools
from functools import reduce
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

from .ems import compute_ems
from .utils import *
from .box import DeformBox
# from ems import compute_ems
# from utils import *
# from box import DeformBox

class DeformContainer(object):
    def __init__(self, length=10, width=10, height=10, rotation=True):
        self.dimension = np.array([length, width, height])
        # heightmap is the map that stores the maximum height of each grid
        self.heightmap = np.zeros(shape=(length, width), dtype=np.int32)
        # stressmap is the map that stores the EQUIVALENT spring constant of each grid
        self.stressmap = np.zeros(shape=(length, width), dtype=np.float32)
        # fragilitymap is the map that stores the MINIMUM fragility index of each grid
        # initialize as 10000 (highest value) since the floor of the box is rigid and can take any weight
        self.fragilitymap = np.ones(shape=(length, width), dtype=np.float32) * 1000
        self.can_rotate = rotation
        # packed box list
        self.boxes = []
        # record rotation information
        self.rot_flags = []
        self.height = height
        self.candidates = [[0, 0, 0]] # TODO: nisara - why is this 3D and not 6D now?

        # map of fragility index to allowed maximum mass on top of it
        # Since the range of fragility index is from 0 to 10 and range of mass is from 1 to 10
        # we can just assume that the fragility index = max mass that package can hold
        self.fragility_mass_map = {}
        for i in range(11):
            self.fragility_mass_map[i] = i

        # The weight_map_3d denotes the total weight ABOVE that grid
        self.weight_map_3d = np.zeros(shape=(length, width, height), dtype=np.float32)
        self.k_map_3d = np.zeros(shape=(length, width, height), dtype=np.float32)
        self.box_id_map_3d = np.zeros(shape=(length, width, height), dtype=np.int32) * -1

        self.deformed_values = []

    # !! not being called anywhere
    def print_heightmap(self):
        print("container heightmap: \n", self.heightmap)

    # !! not being called anywhere
    def print_stressmap(self):
        print("container stressmap: \n", self.stressmap)

    # !! not being called anywhere
    def print_fragilitymap(self):
        print("container fragilitymap: \n", self.fragilitymap)

    # !! not being called anywhere
    def get_heightmap(self):
        """
        get the heightmap for the ideal situation
        Returns:

        """
        plain = np.zeros(shape=self.dimension[:2], dtype=np.int32)
        for box in self.boxes:
            plain, _ = self.update_heightmap(plain, box)
        return plain

    # !! not being called anywhere
    def update_heightmap_vision(self, vision):
        """
        TODO
        Args:
            vision:

        Returns:

        """
        self.heightmap = vision

    
    def update_heightmap(self, plain, box):
        """
        update heightmap
        Args:
            plain:
            box:

        Returns:

        """
        plain = copy.deepcopy(plain)
        le = box.pos_x
        ri = box.pos_x + box.size_x
        up = box.pos_y
        do = box.pos_y + box.size_y
        box = self.post_package_deform(box)
        max_h = np.max(plain[le:ri, up:do])
        # nisara:: COMMENTED OUT THIS - please recheck again
        # max_h = max(max_h, box.pos_z + box.size_z)
        max_h = box.pos_z + box.size_z
        plain[le:ri, up:do] = max_h
        return plain, box
    
    def post_package_deform(self, box):
        # Since we now have deformable objects, we update the heightmap to include their compression
        # IMP: One simplistic assumption - the weight of each box is equally distributed across all the grids it occupies
        # irrespective of where the support beneath it is

        le, ri = box.pos_x, box.pos_x + box.size_x
        up, do = box.pos_y, box.pos_y + box.size_y
        z = box.pos_z
        weight_map = self.weight_map_3d
        k_map = self.k_map_3d
        box_id_map = self.box_id_map_3d
        updated_z = 0
        # Weight across each column = box_mass / (box.size_x * box.size_y)
        weight_per_column = (box.mass / (box.size_x * box.size_y)) # convert mass to kg here
        # Remember that the map is 0-indexed 
        for i in range(le, ri):
            for j in range(up, do):
                # calculate the compression of the box at this grid
                # Set initial box_id as the one at the 0th index and then move upwards
                total_deform = 0
                k = 0
                curr_h = 0
                z = box.pos_z
                box_id = box_id_map[i][j][0]
                while (k <= z) and (box_id != 0):
                    if box_id_map[i][j][k] == box_id:
                        curr_h += 1
                        k += 1
                    else:
                        # Confirm that the weight_values are same across all grids
                        weight_values = weight_map[i][j][k-curr_h:k] # Just note here that weight_values is going to be a pointer to the array, so once you add it to the new weight, THIS WILL CHANGE TOO!
                        k_values = k_map[i][j][k-curr_h:k]
                        # Python is not ready for this yet :)
                        # assert np.isclose(np.sum(weight_values), weight_values[0] * curr_h), "Weight values are not same across all grids"
                        # assert np.isclose(np.sum(k_values), k_values[0] * curr_h), "K values are not same across all grids"
                        weight_map[i][j][k-curr_h:k] = weight_values + weight_per_column
                        deform_val = weight_per_column * (9.8) / k_values[0]
                        # Round up the deform value to the nearest integer
                        deform_val = int(np.round(deform_val))
                        if deform_val >= 1:
                            # Make sure the deform value is not greater than the current height
                            if deform_val > curr_h:
                                deform_val = curr_h
                            total_deform += deform_val
                            # Update all the values above this point to shift down by deform_val amount
                            weight_map[i][j][k-deform_val: k] = -1
                            k_map[i][j][k-deform_val: k] = -1
                            weight_map[i][j][k-deform_val: z-deform_val] = weight_map[i][j][k:z]
                            k_map[i][j][k-deform_val: z-deform_val] = k_map[i][j][k:z]
                            box_id_map[i][j][k-deform_val: z-deform_val] = box_id_map[i][j][k:z]

                            weight_map[i][j][z-deform_val:z] = 0
                            k_map[i][j][z-deform_val:z] = 0
                            box_id_map[i][j][z-deform_val:z] = 0

                            k = k - deform_val
                            z = z - deform_val 
                                                       
                        curr_h = 0
                        box_id = box_id_map[i][j][k]
                # k should ideally be equal to z here
                if k > z:
                    k = z
                updated_z = max(updated_z, k)
        height = box.size_z 
        # Weight map should include weight ON TOP of this grid, not of the box contained in this grid
        # weight_map[i][j][k:k+height] = weight_per_column
        k_map[le:ri, up:do, updated_z:updated_z+height] = box.spring_k
        box_id_map[le:ri, up:do, updated_z:updated_z+height] = box.box_id

        self.weight_map_3d = weight_map
        self.k_map_3d = k_map
        self.box_id_map_3d = box_id_map
        box.pos_z = updated_z

        return box

                    
    def update_stressmap(self, plain_stress, box):
        """
        Calculates the updated stress values based on the formula for springs in series
        if current k value in plain_stress is 0, then new k = box_k
        else new k = 1 / (1/box_k + 1/plain_stress) for each grid in the box's occupancy area
        """
        plain_stress = copy.deepcopy(plain_stress)
        le = box.pos_x
        ri = box.pos_x + box.size_x
        up = box.pos_y
        do = box.pos_y + box.size_y
        box_k = box.spring_k
        region = plain_stress[le:ri, up:do]
        zero_mask = np.abs(region) < 1e-10
        result = np.empty_like(region)
        result[zero_mask] = box_k
        non_zero_mask = ~zero_mask
        result[non_zero_mask] = 1.0 / ((1.0/box_k) + (1.0/region[non_zero_mask]))
        plain_stress[le:ri, up:do] = result
        return plain_stress
    
    def update_fragilitymap(self, plain_fragility, plain_height, box):
        """ 
        Calculates the updated fragility map based on the MINIMUM fragility value of the box and the current fragility map
        """
        # update the fragility map based on the number of boxes on top of it
        # For example, if min fragility is 5, and it already has boxes of 3kgs placed on it
        # Then updated fragility should be 5-3=2
        plain_fragility = copy.deepcopy(plain_fragility)
        le = box.pos_x
        ri = box.pos_x + box.size_x
        up = box.pos_y
        do = box.pos_y + box.size_y
        box_f = box.fragility
        place_z = np.max(plain_height[le:ri, up:do])
        
        # Calculate the valid points (which contain a box) under this box
        points = []
        for x in range(le, ri):
            for y in range(up, do):
                if plain_height[x][y] == (place_z):
                    points.append([x, y])

        # print("Points under this box:", len(points))
        
        if len(points) > 0:
            box_mass_per_grid = np.round(box.mass / len(points))
        else:
            box_mass_per_grid = np.round(box.mass / (box.size_x * box.size_y))
        for point in points:
            # Do a check here to see if the fragility value is less than the box mass
            plain_fragility[point[0]][point[1]] -= box_mass_per_grid
    


        plain_fragility[le:ri, up:do] = np.minimum(plain_fragility[le:ri, up:do], box_f)

        return plain_fragility    
        

    # !! not being called anywhere
    def get_box_list(self):
        vec = list()
        for box in self.boxes:
            vec += box.standardize()
        return vec

    # !! not being called anywhere
    def get_plain(self):
        return copy.deepcopy(self.heightmap)

    # !! not being called anywhere
    def get_action_space(self):
        return self.dimension[0] * self.dimension[1]

    # !! not being called anywhere
    # TODO: this is not being used anywhere - maybe just used for testing? (no EMS)
    def get_action_mask(self, next_box, scheme="heightmap"):
        
        action_mask = np.zeros(shape=(self.dimension[0], self.dimension[1]), dtype=np.int32)

        if scheme == "heightmap":
            candidates_xy, extra_corner_xy = self.candidate_from_heightmap(next_box, self.can_rotate)

            for xy in candidates_xy:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1
            for xy in extra_corner_xy[:3]:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1

            if self.can_rotate:
                rotated_box = [next_box[1], next_box[0], next_box[2]]
                action_mask_rot = np.zeros_like(action_mask)

                for xy in candidates_xy:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1
                for xy in extra_corner_xy[-3:]:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1

                action_mask = np.hstack((action_mask.reshape((-1,)), action_mask_rot.reshape((-1,))))

        elif scheme == "EP": # Extreme Points scheme
            candidates_xy, extra_corner_xy = self.candidate_from_EP(next_box, self.can_rotate)
            # extra_corner_xy = []
            for xy in candidates_xy:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1
            for xy in extra_corner_xy[:3]:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1

            if self.can_rotate:
                rotated_box = [next_box[1], next_box[0], next_box[2]]
                action_mask_rot = np.zeros_like(action_mask)

                for xy in candidates_xy:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1
                for xy in extra_corner_xy[-3:]:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1

                action_mask = np.hstack((action_mask.reshape((-1,)), action_mask_rot.reshape((-1,))))

        elif scheme == "FC":
            # FC considers every single position in the container as a potential placement location
            x_list = list(range(self.dimension[0]))
            y_list = list(range(self.dimension[1]))
            candidates_xy = list(itertools.product(x_list, y_list))

            for xy in candidates_xy:
                if self.check_box(next_box, xy) > -1:
                    action_mask[xy[0], xy[1]] = 1

            if self.can_rotate:
                rotated_box = [next_box[1], next_box[0], next_box[2]]
                action_mask_rot = np.zeros_like(action_mask)

                for xy in candidates_xy:
                    if self.check_box(rotated_box, xy) > -1:
                        action_mask_rot[xy[0], xy[1]] = 1
                
                action_mask = np.hstack((action_mask.reshape((-1,)), action_mask_rot.reshape((-1,))))

            # assert False, 'No FC implementation'
        else:
            assert False, 'Wrong candidate generation scheme'

        # if all actions are invalid, set all mask is 1 and perform any action to end this episode
        if action_mask.sum() == 0:
            action_mask[:] = 1

        return action_mask.reshape(-1).tolist()

    # Even though there exists a check_box_ems function, 
    # this function is called in place_box
    # since without the benchmark flag, both functions are essentially the same
    # nisara: this is where the real check for mass over fragility should be done
    # in is_stable function
    def check_box(self, box_size, pos_xy, box_properties, benchmark=False):
        """
            check
            1. whether cross the border
            2. check stability
        Args:
            box_size: Size of the box that needs to be placed
            pos_xy: Position where the box is to be placed

        Returns:

        """
        if pos_xy[0] + box_size[0] > self.dimension[0] or pos_xy[1] + box_size[1] > self.dimension[1]:
            return -1

        pos_z = np.max(self.heightmap[pos_xy[0]:pos_xy[0] + box_size[0], pos_xy[1]:pos_xy[1] + box_size[1]])

        # whether cross the broder
        if pos_z + box_size[2] > self.dimension[2]:
            return -1
        
        # check stability
        if benchmark:
            # zhao AAAI2021 paper
            rec = self.heightmap[pos_xy[0]:pos_xy[0] + box_size[0], pos_xy[1]:pos_xy[1] + box_size[1]]
            r00 = rec[0, 0]
            r10 = rec[box_size[0] - 1, 0]
            r01 = rec[0, box_size[1] - 1]
            r11 = rec[box_size[0] - 1, box_size[1] - 1]
            rm = max(r00, r10, r01, r11)
            sc = int(r00 == rm) + int(r10 == rm) + int(r01 == rm) + int(r11 == rm)
            # at least 3 support point
            if sc < 3:
                return -1
            # check area and corner
            max_area = np.sum(rec == pos_z)
            area = box_size[0] * box_size[1]
            # 
            if max_area / area > 0.95:
                return pos_z
            if rm == pos_z and sc == 3 and max_area/area > 0.85:
                return pos_z
            if rm == pos_z and sc == 4 and max_area/area > 0.50:
                return pos_z
        else:
            if self.is_stable(box_size, [pos_xy[0], pos_xy[1], pos_z], box_properties):
                return pos_z

        return -1

    def check_box_ems(self, box_size, ems, box_properties, benchmark=False):
        """
            check
            1. whether a box can fit within a given EMS space
            2. whether EMS is within the container if the box starts at the corner points
            3. whether the box crosses the borders of the bin
            4. check stability
        Args:
            box_size: Size of the box that needs to be placed
            pos_xy:

        Returns:

        """
        # Check whether a box can fit within a given EMS space
        if ems[3] - ems[0] < box_size[0] or ems[4] - ems[1] < box_size[1] or ems[5] - ems[2] < box_size[2]:
            return -1

        # Check whether EMS is within the container if the box starts at the corner points
        if ems[0] + box_size[0] > self.dimension[0] or ems[1] + box_size[1] > self.dimension[1]:
            return -1

        pos_z = np.max(self.heightmap[ems[0]:ems[0] + box_size[0], ems[1]:ems[1] + box_size[1]])

        # whether cross the broder
        if pos_z + box_size[2] > self.dimension[2]:
            return -1
        
        # check stability
        if self.is_stable(box_size, [ems[0], ems[1], pos_z], box_properties):
            return pos_z

        return -1

    # conducts a 'physics-based' stability check
    # add check for mass and fragility here
    def is_stable(self, dimension, position, box_properties) -> bool:
        """
            check stability for 3D packing
        Args:
            dimension: Dimension of the box that needs to be placed
            position: Position where the box is to be placed

        Returns:

        """
        # helper function to check if point Q lies on the line segment P1-P2
        def on_segment(P1, P2, Q):
            if ((Q[0] - P1[0]) * (P2[1] - P1[1]) == (P2[0] - P1[0]) * (Q[1] - P1[1]) and
                min(P1[0], P2[0]) <= Q[0] <= max(P1[0], P2[0]) and
                min(P1[1], P2[1]) <= Q[1] <= max(P1[1], P2[1])):
                return True
            else:
                return False

        # item on the ground of the bin, so always 'stable'
        if position[2] == 0:
            return True # no need to check fragility since it is on the floor

        # calculate barycentric coordinates, -1 means coordinate indices start at zero
        # x1, y1 = bottom-left corner of the box base
        # x2, y2 = top-right corner of the box base
        x_1 = position[0]
        x_2 = x_1 + dimension[0] - 1
        y_1 = position[1]
        y_2 = y_1 + dimension[1] - 1
        z = position[2] - 1
        obj_center = ((x_1 + x_2) / 2, (y_1 + y_2) / 2)

        # valid points right under this object
        points = []
        # points_frag = []
        for x in range(x_1, x_2 + 1):
            for y in range(y_1, y_2 + 1):
                if self.heightmap[x][y] == (z + 1):
                    points.append([x, y]) # appends each point 
        # nisara:: COMMENT PRINT
        # self.heightmap_to_compare = self.heightmap
        # self.dimesnion_to_compare = [x_1, x_2, y_1, y_2, z]

        # self.points_to_compare = points

        # the support area is more than half of the bottom surface of the item
        if len(points) > dimension[0] * dimension[1] * 0.5:
            # self.check = "1"
            return self.check_box_fragility(dimension, points, obj_center, box_properties)
        
        if len(points) == 0 or len(points) == 1: 
            return False # no support
        elif len(points) == 2: # whether the center lies on the line of the two points
            if on_segment(points[0], points[1], obj_center):
                # self.check = "2"
                return self.check_box_fragility(dimension, points, obj_center, box_properties)
            else:
                return False # no support
        else:
            # calculate the convex hull of the points
            points = np.array(points)
            try:
                convex_hull = ConvexHull(points)
            except:
                # error means co-lines
                start_p = min(points, key=lambda p: [p[0], p[1]])
                end_p = max(points, key=lambda p: [p[0], p[1]])
                if on_segment(start_p, end_p, obj_center):
                    # self.check = "3"
                    return self.check_box_fragility(dimension, points, obj_center, box_properties)
                else:
                    return False

            hull_path = Path(points[convex_hull.vertices])

            if hull_path.contains_point(obj_center):
                # self.check = "4"
                return self.check_box_fragility(dimension, points, obj_center, box_properties)
            else:
                return False
        
    def check_box_fragility(self, box_size, points, obj_center, box_properties):
        box_mass, _, _ = box_properties 
        
        points_np = np.array(points)
        center_np = np.array(obj_center)
        distances = np.sqrt(np.sum((points_np - center_np) ** 2, axis=1))

        box_mass = np.round(box_mass / len(points_np))

        # # calculate load distance using lever principle
        # # TODO: check this
        # total_load_distance = np.sum(distances)
        # if total_load_distance > 0:
        #     mass_distribution = (distances / total_load_distance) * box_mass
        #     # mass_distribution = mass_distribution * (box_mass / np.sum(mass_distribution))

        #     # Check if any support point exceedds the allowed mass limit
        #     for i, point in enumerate(points):
        #         # print("Comes here!!")
        #         x, y = point
        #         if mass_distribution[i] > self.fragility_mass_map[self.fragilitymap[x, y]]:
        #             return False
        # return True

        # Each point in the support area will have the same mass distribution irrespective of the distance from the center
        # Check if any support point exceeds the allowed mass limit
        # nisara:: COMMENT PRINT
        # self.points_np_to_compare = points_np
        # self.box_mass_to_compare = box_mass
        # self.fragilitymap_to_compare = self.fragilitymap
        for point in points_np:
            x, y = point
            # print("value: ", self.fragilitymap[x, y])
            # print("Box mass: ", box_mass)
            
            if self.fragilitymap[x, y] < box_mass:
                return False
        return True
    

    def get_volume_ratio(self):
        # Ration of the volume of packed boxes to the volume of the container
        # vo = reduce(lambda x, y: x + y, [box.size_x * box.size_y * box.size_z for box in self.boxes], 0.0)
        # Calculate volume by seeing the heightmap - the height of each grid is given by the heightmap
        vo = np.sum(self.heightmap)
        mx = self.dimension[0] * self.dimension[1] * self.dimension[2]
        ratio = vo / mx
        assert ratio <= 1.0
        if ratio < 0:
            print("VOLUME RATIO IS NEGATIVE: ", ratio)
        return ratio

    # 1d index -> 2d plain coordinate
    def idx_to_position(self, idx):
        """
        TODO
        Args:
            idx:

        Returns:

        """
        lx = idx // self.dimension[1]
        ly = idx % self.dimension[1]
        return lx, ly

    def position_to_index(self, position):
        assert len(position) == 2
        assert position[0] >= 0 and position[1] >= 0
        assert position[0] < self.dimension[0] and position[1] < self.dimension[1]
        return position[0] * self.dimension[1] + position[1]

    def place_box(self, box_size, pos, rot_flag, properties):
        """ place box in the position (index), then update heightmap
        :param box_size: length, width, height
        :param idx:
        :param rot_flag:
        :return:
        """
        if not rot_flag:
            size_x = box_size[0]
            size_y = box_size[1]
        else:
            size_x = box_size[1]
            size_y = box_size[0]
        size_z = box_size[2]
        plain = self.heightmap
        # Ideally, check for mass and fragility here in check_box but 
        # since place_box is called in the final stages of placement
        # it should be checked while getting candidate ems
        new_h = self.check_box([size_x, size_y, size_z], [pos[0], pos[1]], properties) 
        if new_h != -1:
            # nisara: changed to deform box here and added the required properties
            mass, spring_k, fragility = properties[0], properties[1], properties[2]
            box_id_index = len(self.boxes) + 1
            self.boxes.append(DeformBox(box_id_index, size_x, size_y, size_z, pos[0], pos[1], new_h, mass=mass, spring_k=spring_k, fragility=fragility))  # record rotated box
            self.rot_flags.append(rot_flag)
            self.deformed_values.append(size_z)
            # POST-PACKAGE DEFORMATION is added before updating the heightmap
            self.fragilitymap = self.update_fragilitymap(self.fragilitymap, self.heightmap, self.boxes[-1])
            self.heightmap, self.boxes[-1] = self.update_heightmap(plain, self.boxes[-1])
            self.height = max(self.height, self.boxes[-1].pos_z + size_z)
            self.stressmap = self.update_stressmap(self.stressmap, self.boxes[-1])
            
            return True
        return False

    def candidate_from_heightmap(self, next_box, box_properties, max_n) -> list:
        """
        get the x and y coordinates of candidates
        Args:
            next_box:
            can_rotate:

        Returns:

        """
        # More like a Corner Points implementation
        
        heightmap = copy.deepcopy(self.heightmap)

        corner_list = []
        # hm_diff: height differences of neighbor columns, padding 0 in the front
        # x coordinate -> r represents row
        # heightmap: [r0, r1, r2, r3, r4, r5, ..., rn]
        # insert: [r0, r0, r1, r2, r3, r4, r5, ..., rn]
        hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
        # delete: [r0, r0, r1, r2, r3, r4, ..., rn-1]
        hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x) - 1, axis=0)
        # hm_diff_x: [0, r1-r0, r2-r1, r3-r2, r4-r3, r5-r4, rn-r(n-1)]
        hm_diff_x = heightmap - hm_diff_x

        # y coordinate
        hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
        hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T) - 1, axis=1)
        # hm_diff_y: [0, c1-c0, c2-c1, c3-c2, c4-c3, c5-c4, cn-c(n-1)]
        hm_diff_y = heightmap - hm_diff_y

        # get the xy coordinates of all left-deep-bottom corners
        # Non-zero values in these arrays indicate "steps" or "corners" in the heightmap surface
        corner_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
        corner_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()

        corner_xy_list = []
        corner_xy_list.append([0, 0])

        """
        For a box with dimensions (w,h,d), 
        if point (x,y) is valid and point (x,y+1) has the same height, 
        placing the box at either position would result in nearly identical configurations - 
        the only difference would be 1 unit of shift along the y-axis
        """
        for xy in corner_x_list:
            x, y = xy
            if y != 0 and [x, y - 1] in corner_x_list:
                # if heightmap[x, y] == heightmap[x, y - 1] and hm_diff_x[x, y] == hm_diff_x[x, y - 1]:
                if heightmap[x, y] == heightmap[x, y - 1]:
                    continue
            corner_xy_list.append(xy)
        for xy in corner_y_list:
            x, y = xy
            if x != 0 and [x - 1, y] in corner_y_list:
                # if heightmap[x, y] == heightmap[x - 1, y] and hm_diff_x[x, y] == hm_diff_x[x - 1, y]:
                if heightmap[x, y] == heightmap[x - 1, y]:
                    continue
            if xy not in corner_xy_list:
                corner_xy_list.append(xy)

        candidate_x, candidate_y = zip(*corner_xy_list)
        # remove duplicate elements
        candidate_x = list(set(candidate_x))
        candidate_y = list(set(candidate_y))

        # get corner_list
        corner_list = list(itertools.product(candidate_x, candidate_y))
        candidates = [] 

        for xy in corner_list:
            z = self.check_box(next_box, xy)
            if z > -1:
                # candidates.append([xy[0], xy[1], z, 0])
                candidates.append([xy[0], xy[1], z, xy[0] + next_box[0], xy[1] + next_box[1], z + next_box[2]])
        
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for xy in corner_list:
                z = self.check_box(rotated_box, xy)
                if z > -1:
                    # candidates.append([xy[0], xy[1], z, 1])
                    candidates.append([xy[0], xy[1], z, xy[0] + rotated_box[0], xy[1] + rotated_box[1], z + rotated_box[2]])

        # sort by z, y coordinate, then x
        candidates.sort(key=lambda x: [x[2], x[1], x[0]])

        if len(candidates) > max_n:
            candidates = candidates[:max_n]
        self.candidates = candidates
        return np.array(candidates)

    def candidate_from_EP(self, next_box, box_properties, max_n) -> list:
        """
        calculate extreme points from items extracted from current heightmap
        Args:
            new_item:

        Returns:

        """
        heightmap = copy.deepcopy(self.heightmap)
        items_in = extract_items_from_heightmap(heightmap)
        new_eps = []
        new_eps.append([0, 0, 0])

        for k in range(len(items_in)):
            items_in_copy = copy.deepcopy(items_in)
            item_new = items_in_copy[k]
            new_dim = item_new[:3]
            new_pos = item_new[-3:]

            items_in_copy.pop(k)
            item_fitted = items_in_copy

            # add xoy, xoz, yoz planes for easy projection
            item_fitted.append([self.dimension[0], self.dimension[1], 0, 0, 0, 0])
            item_fitted.append([self.dimension[0], 0, self.dimension[2], 0, 0, 0])
            item_fitted.append([0, self.dimension[1], self.dimension[2], 0, 0, 0])

            max_bounds = [-1, -1, -1, -1, -1, -1]

            for i in range(len(item_fitted)):
                fitted_dim = item_fitted[i][:3]
                fitted_pos = item_fitted[i][-3:]
                project_x = fitted_dim[0] + fitted_pos[0]
                project_y = fitted_dim[1] + fitted_pos[1]
                project_z = fitted_dim[2] + fitted_pos[2]

                # Xy - new_eps[0]
                if can_take_projection(item_new, item_fitted[i], 0, 1) and project_y > max_bounds[Projection.Xy]:
                    new_eps.append([new_pos[0] + new_dim[0], project_y, new_pos[2]])
                    max_bounds[Projection.Xy] = project_y

                # Xz - new_eps[1]
                if can_take_projection(item_new, item_fitted[i], 0, 2) and project_z > max_bounds[Projection.Xz]:
                    new_eps.append([new_pos[0] + new_dim[0], new_pos[1], project_z])
                    max_bounds[Projection.Xz] = project_z

                # Yx - new_eps[2]
                if can_take_projection(item_new, item_fitted[i], 1, 0) and project_x > max_bounds[Projection.Yx]:
                    new_eps.append([project_x, new_pos[1] + new_dim[1], new_pos[2]])
                    max_bounds[Projection.Yx] = project_x

                # Yz - new_eps[3]
                if can_take_projection(item_new, item_fitted[i], 1, 2) and project_z > max_bounds[Projection.Yz]:
                    new_eps.append([new_pos[0], new_pos[1] + new_dim[1], project_z])
                    max_bounds[Projection.Yz] = project_z

                # Zx - new_eps[4]
                if can_take_projection(item_new, item_fitted[i], 2, 0) and project_x > max_bounds[Projection.Zx]:
                    new_eps.append([project_x, new_pos[1], new_pos[2] + new_dim[2]])
                    max_bounds[Projection.Zx] = project_x

                # Zy - new_eps[5]
                if can_take_projection(item_new, item_fitted[i], 2, 1) and project_y > max_bounds[Projection.Zy]:
                    new_eps.append([new_pos[0], project_y, new_pos[2] + new_dim[2]])
                    max_bounds[Projection.Zy] = project_y

        new_eps = [ep for ep in new_eps if not (ep[0] == self.dimension[0] or 
                                                ep[1] == self.dimension[1] or 
                                                ep[2] == self.dimension[2])]

        # only need x, y
        new_eps = np.array(new_eps, dtype=np.int32)

        # remove duplicates
        new_eps = np.unique(new_eps, axis=0)
        candidates = new_eps.tolist()
        candidates.sort(key=lambda x: [x[2], x[1], x[0]])
        mask = np.zeros((2, max_n), dtype=np.int8)

        if len(candidates) > max_n:
            candidates = candidates[:max_n]

        for id, ep in enumerate(candidates):
            z = self.check_box(next_box, ep)
            if z > -1 and z == ep[2]:
                mask[0, id] = 1 
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for id, ep in enumerate(candidates):
                z = self.check_box(rotated_box, ep)
                if z > -1 and z == ep[2]:
                    mask[1, id] = 1 

        self.candidates = candidates
        return np.array(candidates), mask
    
    # !! focus on this function since we use EMS primarily
    def candidate_from_EMS(self, 
        next_box, 
        box_properties,
        max_n
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ calculate Empty Maximum Space from items extracted from current heightmap

        Args:
            next_box to be placed
            box_properties: properties of the box to place (mass, k, fragility_index)
            max_n: maximum number of candidate placements to return

        Returns:
            list: 
        """
        heightmap = copy.deepcopy(self.heightmap)
        # left-bottom & right-top pos [bx, by, bz, tx, ty, tz], 
        # dimension: [tx-bx, ty-by, tz-bz],
        all_ems = compute_ems(heightmap, container_h=self.dimension[2])  

        # print("EMS: ", all_ems)

        candidates = all_ems
        # size is 2 since we need to check for both orientations of the box
        mask = np.zeros((2, max_n), dtype=np.int8)
        
        # sort by z, y coordinate, then x
        # candidates = list of these elements: [x_small, y_small, h, x_large, y_large, container_h]
        candidates.sort(key=lambda x: [x[2], x[1], x[0]])

        if len(candidates) > max_n:
            candidates = candidates[:max_n]
        
        for id, ems in enumerate(candidates):
            if self.check_box_ems(next_box, ems, box_properties=box_properties) > -1:
                mask[0, id] = 1
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for id, ems in enumerate(candidates):
                if self.check_box_ems(rotated_box, ems, box_properties=box_properties) > -1:
                    mask[1, id] = 1
        
        self.candidates = candidates
        # dimensions of candidates - (num_candidates, 6)
        return np.array(candidates), mask

    def candidate_from_FC(self, next_box, box_properties) -> list:
        """
        calculate extreme points from items extracted from current heightmap
        Args:
            new_item:

        Returns:

        """
        candidates = []

        for x in range(self.dimension[0]):
            for y in range(self.dimension[1]):
                candidates.append([x, y, self.heightmap[x, y]])

        mask = np.zeros((2, self.dimension[0]*self.dimension[1]), dtype=np.int8)

        for id, xyz in enumerate(candidates):
            z = self.check_box(next_box, xyz)
            if z > -1 and z == xyz[2]:
                mask[0, id] = 1 
        if self.can_rotate:
            rotated_box = [next_box[1], next_box[0], next_box[2]]
            for id, xyz in enumerate(candidates):
                z = self.check_box(rotated_box, xyz)
                if z > -1 and z == xyz[2]:
                    mask[1, id] = 1 

        self.candidates = candidates
        return np.array(candidates), mask


# if __name__ == '__main__':
#     container = DeformContainer(5, 5, 10)

#     container.print_fragilitymap()
#     container.print_stressmap()
#     container.print_heightmap()

#     next_box = [2, 3, 4]
#     next_box_props = np.array([6.0, 7.6, np.round(7.6)])
#     candidates, mask = container.candidate_from_EMS(next_box, np.array(next_box_props), 100)
#     print("Candidates and mask: ", candidates, mask)
#     placed = container.place_box(next_box, np.array([0,0,0]), 0, next_box_props)
#     print("Placed: ", placed)
#     container.print_fragilitymap()
#     container.print_stressmap()
#     container.print_heightmap()

#     next_box = [2, 3, 4]
#     next_box_props = np.array([3.0, 4.9, np.round(4.9)])
#     candidates, mask = container.candidate_from_EMS(next_box, np.array(next_box_props), 100)
#     print("Candidates and mask: ", candidates, mask)
#     placed = container.place_box(next_box, np.array([0,0,4]), 0, next_box_props)
#     print("Placed: ", placed)
#     container.print_fragilitymap()
#     container.print_stressmap()
#     container.print_heightmap()

#     next_box = [1, 4, 2]
#     next_box_props = np.array([4.0, 3.9, np.round(3.9)])
#     candidates, mask = container.candidate_from_EMS(next_box, np.array(next_box_props), 100)
#     print("Candidates and mask: ", candidates, mask)
#     placed = container.place_box(next_box, np.array([0,0,7]), 0, next_box_props)
#     print("Placed: ", placed)
#     container.print_fragilitymap()
#     container.print_stressmap()
#     container.print_heightmap()

#     next_box = [1, 4, 2]
#     next_box_props = np.array([6.0, 3.9, np.round(3.9)])
#     candidates, mask = container.candidate_from_EMS(next_box, np.array(next_box_props), 100)
#     print("Candidates and mask: ", candidates, mask)
#     placed = container.place_box(next_box, np.array([1,0,7]), 0, next_box_props)
#     print("Placed: ", placed)
#     container.print_fragilitymap()
#     container.print_stressmap()
#     container.print_heightmap()

