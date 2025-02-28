import os
import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce

from parameter import *
from classes.utils import *
from classes.env.sensor import sensor_work


class Env:
    def __init__(self, episode_index, n_agent=1, plot=False, test=False):
        self.episode_index = episode_index
        self.plot = plot
        self.test = test
        self.n_agent = n_agent

        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)  # meter

        self.sensor_range = SENSOR_RANGE  # meter
        self.explored_rate = 0

        self.done = False

        self.robot_belief = sensor_work(initial_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        free, _ = get_updating_node_coords(np.array([0.0, 0.0]), self.belief_info)
        choice = np.random.choice(free.shape[0], self.n_agent, replace=False)
        if self.test: # fix the start position if it is test
            self.robot_locations = np.array([[0.0, 0.0] * self.n_agent])
        else:
            starts = free[choice]
            self.robot_locations = np.array(starts)

        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info).reshape(-1, 2)
        for robot_cell in robot_cells:
            self.robot_belief = sensor_work(robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                            self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)
        self.global_frontiers = get_frontier_in_map(self.belief_info)
        self.old_frontiers = deepcopy(self.global_frontiers)

        if self.plot:
            self.frame_files = []

        self.expert_planner = None
        self.ground_truth_planner = None

    def import_ground_truth(self, episode_index):
        if self.test:
            map_dir = f'maps_test'
        else:
            map_dir = f'maps_train'
        map_list = sorted(os.listdir(map_dir))
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1)).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.array(np.nonzero(ground_truth == 208))
        robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell):
        self.robot_belief = sensor_work(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self):
        reward = 0

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def check_done(self):
        if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
            self.done = True

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, agent_id):
        self.evaluate_exploration_rate()
        self.robot_locations[agent_id] = next_waypoint
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.update_robot_belief(cell)

    def get_expert_paths(self):
        paths = self.expert_planner.plan_coverage_paths(self.robot_locations)
        return paths

    def get_ground_truth_paths(self):
        self.global_frontiers = get_frontier_in_map(self.belief_info)
        changed_frontiers = self.old_frontiers ^ self.global_frontiers
        delta_num = len(changed_frontiers)
        if delta_num == 0:
            map_change = False
        else:
            map_change = True
        self.old_frontiers = deepcopy(self.global_frontiers)
        paths = self.ground_truth_planner.plan_coverage_paths(self.belief_info, self.robot_locations, map_change)
        return paths

