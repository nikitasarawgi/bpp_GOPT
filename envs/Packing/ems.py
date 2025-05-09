
import copy
import itertools
import time

import numpy as np


def compute_corners(heightmap: np.ndarray):
    # Corner detection from heightmap by udentifying locations where 
    # the height changes in both x and y directions

    hm_shape = heightmap.shape # shape : [x_dim, y_dim]
    extend_hm = np.ones((hm_shape[0]+2, hm_shape[1]+2)) * 10000 # Pad the heightmap and then assign them really high values - to simulate the walls (in all 4 directions)
    extend_hm[1:-1, 1:-1] = heightmap # Put the original heightmap in the middle of the padded heightmap

    x_diff_hm_1 = extend_hm[:-1] - extend_hm[1:]
    x_diff_hm_1 = x_diff_hm_1[:-1, 1:-1]  # the slicing operation to remove the paddings that we had earlier

    x_diff_hm_2 = extend_hm[1:] - extend_hm[:-1]
    x_diff_hm_2 = x_diff_hm_2[1:, 1:-1]  

    y_diff_hm_1 = extend_hm[:, :-1] - extend_hm[:, 1:]
    y_diff_hm_1 = y_diff_hm_1[1:-1, :-1] 

    y_diff_hm_2 = extend_hm[:, 1:] - extend_hm[:, :-1]
    y_diff_hm_2 = y_diff_hm_2[1:-1, 1:]  
    
    # create a list of boolean arrays of whether height changes or not
    # x_diff_hm_1 != 0: Height drops when moving in the +x direction (right)
    # x_diff_hm_2 != 0: Height rises when moving in the +x direction (right)
    # y_diff_hm_1 != 0: Height drops when moving in the +y direction (down)
    # y_diff_hm_2 != 0: Height rises when moving in the +y direction (down)
    x_diff_hms = [x_diff_hm_1 != 0, x_diff_hm_2 != 0] 
    y_diff_hms = [y_diff_hm_1 != 0, y_diff_hm_2 != 0]

    # A corner exists where height changes simultaneously in both x and y directions
    corner_hm = np.zeros_like(heightmap)
    for xhm in x_diff_hms:
        for yhm in y_diff_hms:
            corner_hm += xhm * yhm # Multiplication of True and False 

    # EMS spaces are represented by the left-bottom corner of the container
    # and the right-top corner of the container
    # the top-right corner can be easily calculated using the left-bottom corner if we know the size of the ems space
    left_bottom_hm = (x_diff_hm_1 != 0) * (y_diff_hm_1 != 0)

    left_bottom_corners = np.where(left_bottom_hm > 0)
    # transpose just to change the shape of the array to (2, n) where n is the number of corners
    left_bottom_corners = np.array(left_bottom_corners).transpose()

    corners = np.where(corner_hm > 0)
    corners = np.array(corners).transpose()

    # x_borders = list(np.where(x_diff_hm_1.sum(axis=1))[0])
    # y_borders = list(np.where(y_diff_hm_1.sum(axis=0))[0])
    # TODO: BUG: should this not be [1] for y_borders?
    x_borders = list(np.unique(np.where(x_diff_hm_1 != 0)[0]))
    y_borders = list(np.unique(np.where(y_diff_hm_1 != 0)[0]))
    
    # The bin container sides are also borders
    x_borders.append(hm_shape[0])
    y_borders.append(hm_shape[1])

    return corners, left_bottom_corners, x_borders, y_borders


def compute_stair_corners(heightmap, corners):

    corners, _, _, _ = compute_corners(heightmap)

    stair_hm = np.zeros_like(heightmap)
    corner_heights = heightmap[corners[:,0], corners[:,1]]
    sort_ids = np.argsort(corner_heights)
    # sort the corners based on their heights
    sort_corners = corners[sort_ids]

    for c in sort_corners:
        cx, cy = c
        h = heightmap[cx, cy]
        # fills the entire rectangle from (0,0) to (cx,cy) with this height
        stair_hm[:cx+1, :cy+1] = h
    
    # only use the left-bottom corners this time
    _, slb_corner, _, _ = compute_corners(stair_hm)
    return slb_corner


def compute_empty_space(
        container_h, 
        corners, 
        x_borders, 
        y_borders, 
        heightmap, 
        empty_space_list, # store the empty spaces here
        x_side='left-right', 
        y_side='left-right', 
        min_ems_width=0, 
        container_id=0
    ):
    # NOTE find ems from corners
    # EMS: [ [bx,by,bz], [tx,ty,tz], [i,i,i] ]
    #   1. left-bottom pos [bx, by, bz]
    #   2. right-top pos: [tx, ty, tz]
    #   3. container_id: [i, i, i]
    
    def check_valid_height_layer(height_layer):
        return (height_layer <= 0).all()

    # check for each corner as a potential starting point for any empty space
    for corner in corners:
        x,y = corner
        # h = int(heightmap[x, y])
        h = heightmap[x, y]
        if h == container_h: continue

        # the WHOLE heightmap is shifted down
        h_layer = heightmap - h

        for axes in itertools.permutations(range(2), 2):
            x_small = x
            x_large = x+1
            
            y_small = y
            y_large = y+1

            for axis in axes:
                if axis == 0:
                    if 'left' in x_side:
                        for xb in x_borders:
                            if x_small > xb:
                                # if (h_layer[xb:x, y_small:y_large] <= 0).all():
                                if check_valid_height_layer(h_layer[xb:x, y_small:y_large]):
                                    x_small = xb
                            else: break

                    if 'right' in x_side:
                        for xb in x_borders[::-1]:
                            if x_large < xb:
                                if check_valid_height_layer(h_layer[x:xb, y_small:y_large]):
                                # if (h_layer[x:xb, y_small:y_large] <= 0).all():
                                    x_large = xb
                            else: break
                
                elif axis == 1:
                    if 'left' in y_side:
                        for yb in y_borders:
                            if y_small > yb:
                                if check_valid_height_layer(h_layer[ x_small:x_large, yb:y]):
                                # if (h_layer[ x_small:x_large, yb:y] <= 0).all():
                                    y_small = yb
                            else: break

                    if 'right' in y_side:
                        for yb in y_borders[::-1]:
                            if y_large < yb:
                                if check_valid_height_layer(h_layer[ x_small:x_large, y:yb]):
                                # if (h_layer[ x_small:x_large, y:yb] <= 0).all():
                                    y_large = yb
                            else: break

            # if (h_layer[ x_small:x_large, y_small:y_large] <= 0).all():
            if check_valid_height_layer(h_layer[x_small:x_large, y_small:y_large]):

                # new_ems = [[x_small, y_small, h], [x_large, y_large, container_h],[container_id]*3 ]
                new_ems = [x_small, y_small, h, x_large, y_large, container_h]

                if (x_large - x_small <= 0) or (y_large - y_small <= 0) :
                    new_ems = None

                # NOTE remove small ems
                if min_ems_width > 0:
                    if x_large - x_small < min_ems_width or y_large - y_small < min_ems_width:
                        new_ems = None

                if new_ems is not None and new_ems not in empty_space_list:
                    empty_space_list.append(new_ems)

def compute_ems(
        heightmap: np.ndarray, 
        container_h: int, 
        min_ems_width: int = 0, 
        id_map: np.ndarray = None
    ) -> list:
    container_h = int(container_h)
    empty_max_spaces = []
    
    if id_map is not None:
        m = id_map
    else:
        m = heightmap
    corners, left_bottom_corners, x_borders, y_borders = compute_corners(m)

    compute_empty_space(
        container_h, 
        left_bottom_corners, 
        x_borders, 
        y_borders, 
        heightmap, 
        empty_max_spaces, 
        'right', 
        'right', 
        min_ems_width=min_ems_width
    )

    compute_empty_space(
        container_h, 
        corners, 
        x_borders, 
        y_borders, 
        heightmap, 
        empty_max_spaces, 
        'left-right', 
        'left-right', 
        min_ems_width=min_ems_width
    )

    # NOTE stair corners
    stair_corners = compute_stair_corners(heightmap, corners)
    compute_empty_space(
        container_h, 
        stair_corners, 
        x_borders, 
        y_borders, 
        heightmap, 
        empty_max_spaces, 
        'right', 
        'right', 
        min_ems_width=min_ems_width
    )
    
    return empty_max_spaces


def add_box(heightmap, box, pos):
    bx, by, bz = box
    px, py, pz = pos
    
    z = heightmap[px: px+bx, py:py+by].max()
    heightmap[px: px+bx, py:py+by] = z + bz


if __name__ == '__main__':
    length = 10
    h = np.zeros([length, length])
    
    # add_box(h, [2,2,1], [0,0,0])
    # add_box(h, [2,2,3], [2,3,0])
    # add_box(h, [2,6,3], [7,3,0])
    # add_box(h, [4,6,7], [0,3,0])
    # add_box(h, [4,6,1], [3,0,0])
    # add_box(h, [4,2,2], [5,2,0])
    add_box(h, [9,9,9], [0,0,0])
    print(h)
    all_ems = compute_ems(h, length)
    # all_ems = compute_ems(np.array(state), 30)
    for ems in all_ems:
        print(ems)
