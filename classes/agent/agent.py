import copy
import time

import torch
import numpy as np

from parameter import *
from classes.utils import *
from classes.agent.node_manager import NodeManager


class Agent:
    def __init__(self, id, node_manager, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.plot = plot

        # location and global map
        self.location = None
        self.map_info = None

        # map related parameters
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = node_manager

        # graph
        self.node_coords, self.utility, self.guidepost= None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        self.travel_dist = 0

        self.episode_buffer = []
        for i in range(15):
            self.episode_buffer.append([])

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []


    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        if self.location is None:
            dist = 0
        else:
            dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location

        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        else:
            node.data.set_visited()

        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[
                                  0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                  1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_graph(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)

    def update_planning_state(self, robot_locations):
        # self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
        self.node_coords, self.utility, self.guidepost, self.occupancy, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
            self.node_manager.get_all_node_graph(self.location, robot_locations)

    def get_observation(self):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        node_occupancy = self.occupancy.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                             node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, node_occupancy), axis=1)
        # node_inputs = np.concatenate((node_coords, node_utility, node_guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)

    def save_done(self, done):
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation):
        self.episode_buffer[9] = copy.deepcopy(self.episode_buffer[0])[1:]
        self.episode_buffer[10] = copy.deepcopy(self.episode_buffer[1])[1:]
        self.episode_buffer[11] = copy.deepcopy(self.episode_buffer[2])[1:]
        self.episode_buffer[12] = copy.deepcopy(self.episode_buffer[3])[1:]
        self.episode_buffer[13] = copy.deepcopy(self.episode_buffer[4])[1:]
        self.episode_buffer[14] = copy.deepcopy(self.episode_buffer[5])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()

    def get_no_padding_observation(self):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        node_occupancy = self.occupancy.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                            node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate(
            (node_coords, node_utility, node_guidepost, node_occupancy), axis=1)
            # (node_coords, node_utility, node_guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge,
                edge_padding_mask]

