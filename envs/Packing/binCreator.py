import numpy as np
import copy
import torch
import math
import csv
import ast


class BoxCreator(object):
    def __init__(self):
        self.box_list = []  # generated box list

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        """
        :param length:
        :return: list
        """
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)


class RandomDeformBoxCreator(BoxCreator):
    default_box_set = [] # list of tuples to represent possible box dimensions + mass + spring constant + fragility
    # the range of box dimensions is from 2 to 5 (inclusive)
    # the range of mass is from 1 to 10 (inclusive) kg
    # the range of spring constant is 1 to 10 (inclusive)
    # the range of fragility index is 0 to 10 (inclusive) (let's assume fragility index == mass)
    # NOTE: fragility index is the max weight THAT each grid of that box can hold
    for l in range(3):
        for w in range(3):
            for h in range(3):
                for m in range(1, 11, 1):
                    for k in range(1, 11, 1):
                        default_box_set.append((2 + l, 2 + w, 2 + h, m, k, np.round(m)))

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomDeformBoxCreator.default_box_set
        # print(self.box_set)
        print("Box set size: ", len(self.box_set))
        # Load the test indices and keep if needed from the csv file
        self.load_test_indices()
        
    def load_test_indices(self):
        # file_path = "test_indices_4.txt"
        file_path = "video_test_indices_part4.txt"
        self.test_indices = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    self.test_indices.append(ast.literal_eval(line.strip()))
        except Exception as e:
            print(f"Error loading test indices from {file_path}: {e}")
        self.box_index = 0

    def generate_box_size(self, use_test_data = False, test_index = 0, **kwargs):
        if use_test_data: 
            row = self.test_indices[test_index]
            idx = row[self.box_index]
            self.box_index += 1
            print("Test index: ", test_index,  "and index from test data: ", idx)
            print("Box dimensions: ", self.box_set[idx])
        else:
            idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])
    
    def reset(self):
        super().reset()
        self.box_index = 0


class RandomBoxCreator(BoxCreator):
    default_box_set = [] # list of tuples to represetn possible box dimensions
    for i in range(4):
        for j in range(4):
            for k in range(4):
                default_box_set.append((2 + i, 2 + j, 2 + k)) # ranges from 2 to 5 each (inclusive)

    def __init__(self, box_size_set=None):
        super().__init__()
        self.box_set = box_size_set
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set
        # print(self.box_set)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])


# load data
class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):  # data url
        super().__init__()  
        self.data_name = data_name
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))  
        print("load data set successfully, data name: ", self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index # index of which trajectory to follow
        self.boxes = box_trajs[self.index]
        self.box_index = 0 # index of which box within that trajectory
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1
