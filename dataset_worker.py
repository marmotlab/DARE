import os
import time
import random

from copy import deepcopy

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset_parameter import *
from classes.utils import *
from classes.env.env import Env
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager
from classes.planner.expert_planner import ExpertPlanner
from classes.planner.ground_truth_planner import GroundTruthPlanner

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DatasetWorker:
    def __init__(self, meta_agent_id, global_step, device='cpu', save_image=False, greedy=False):
        assert TEST_N_AGENTS == 1, "Only 1 agent is supported for now"
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.greedy = greedy

        self.env = Env(global_step, n_agent=TEST_N_AGENTS, plot=self.save_image, test=USE_TEST_DATASET)
        self.node_manager = NodeManager(plot=self.save_image)

        self.robot_list = [Agent(i, self.node_manager, self.device, self.save_image) for i in
                           range(self.env.n_agent)]

        self.episode_data = dict()
        self.perf_metrics = dict()

    def run_episode(self):
        unique_seed = int(time.time())
        set_random_seed(unique_seed)
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        action_list = []
        episode_end_list = []
        if DATA_TYPE == 'node':
            node_inputs_list = []
            node_padding_mask_list = []
            edge_mask_list = []
            current_index_list = []
            current_edge_list = []
            edge_padding_mask_list = []
        elif DATA_TYPE == 'map':
            img_list = []
            state_list = []
        else:
            raise ValueError('Invalid dataset type, check dataset_parameter.py')

        if DATASET_METHOD == 'tare':
            self.env.expert_planner = ExpertPlanner(self.node_manager)
        if DATASET_METHOD == 'ground_truth':
            self.env.ground_truth_planner = GroundTruthPlanner(self.env.ground_truth_info, robot.node_manager)
        if DATASET_METHOD == 'ground_truth_no_replan':
            self.env.ground_truth_planner = GroundTruthPlanner(self.env.ground_truth_info, robot.node_manager)
            paths = self.env.get_ground_truth_paths()

        for step in range(MAX_EPISODE_STEP): 
            selected_locations = []
            dist_list = []
            
            # HACK: for now, only 1 agent
            if DATA_TYPE == 'node':
                observation = self.robot_list[0].get_observation()
                node_inputs_list.append(observation[0])
                node_padding_mask_list.append(observation[1])
                edge_mask_list.append(observation[2])
                current_index_list.append(observation[3])
                current_edge_list.append(observation[4])
                edge_padding_mask_list.append(observation[5])
            elif DATA_TYPE == 'map':
                img_list.append(deepcopy(self.env.robot_belief))
                state_list.append(deepcopy(self.env.robot_locations[0]))
    
            if DATASET_METHOD == 'tare':
                paths = self.env.get_expert_paths()
                for i, path in enumerate(paths):
                    selected_locations.append(np.array(path[0]))
                    if USE_DELTA_POSITION:
                        action_list.append(np.array(path[0] - self.env.robot_locations[i]))
                    else:
                        action_list.append(np.array(path[0]))
                dist_list = [k for k in range(self.env.n_agent)]

            if DATASET_METHOD == 'ground_truth': 
                paths = self.env.get_ground_truth_paths()
                for i, path in enumerate(paths):
                    selected_locations.append(np.array(path[0]))
                    if USE_DELTA_POSITION:
                        action_list.append(np.array(path[0] - self.env.robot_locations[i]))
                    else:
                        action_list.append(np.array(path[0]))
                dist_list = [k for k in range(self.env.n_agent)]

            if DATASET_METHOD == 'ground_truth_no_replan':
                for i, path in enumerate(paths):
                    selected_locations.append(np.array(path[0]))
                    if USE_DELTA_POSITION:
                        action_list.append(np.array(path[0] - self.env.robot_locations[i]))
                    else:
                        action_list.append(np.array(path[0]))
                    path.pop(0)
                dist_list = [k for k in range(self.env.n_agent)]

            if self.save_image:
                planned_paths = None
                if DATASET_METHOD == "ground_truth" or DATASET_METHOD == "tare" or DATASET_METHOD == "ground_truth_no_replan":
                    planned_paths = paths
                self.plot_env(step, planned_paths)

            # check if the selected locations are the same FOR future multi-agent
            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:,
                                                                          0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            for robot, next_location in zip(self.robot_list, selected_locations):
                self.env.step(next_location, robot.id)

                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))

            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            if self.robot_list[0].utility.sum() == 0:
                done = True

            if done or (DATASET_METHOD == 'ground_truth_no_replan' and len(paths[0]) == 0):
                if self.save_image:
                    self.plot_env(step+1)
                episode_end_list.append(step + 1)
                break

        # save episode data
        self.episode_data['action'] = np.array(action_list, dtype=np.float32)
        self.episode_data['episode_end'] = np.array(episode_end_list, dtype=np.int32)
        if DATA_TYPE == 'node':
            self.episode_data['node_inputs'] = np.squeeze(np.array(node_inputs_list, dtype=np.float32), axis=1)
            self.episode_data['node_padding_mask'] = np.squeeze(np.array(node_padding_mask_list, dtype=np.int16), axis=1)
            self.episode_data['edge_mask'] = np.squeeze(np.array(edge_mask_list, dtype=np.int16), axis=1)
            self.episode_data['current_index'] = np.squeeze(np.array(current_index_list, dtype=np.int32), axis=1)
            self.episode_data['current_edge'] = np.squeeze(np.array(current_edge_list, dtype=np.int32), axis=1)
            self.episode_data['edge_padding_mask'] = np.squeeze(np.array(edge_padding_mask_list, dtype=np.int16), axis=1)
        elif DATA_TYPE == 'map':
            self.episode_data['img'] = np.array(img_list, dtype=np.float32)
            self.episode_data['state'] = np.array(state_list, dtype=np.float32)
        else:
            raise ValueError('Invalid dataset type, check dataset_parameter.py')
        
        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_env(self, step, planned_paths=None):
        self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y']
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)
                plt.imshow(robot.map_info.map, cmap='gray')
                plt.axis('off')
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)
                for node, utility in zip(nodes, robot.utility):
                    plt.text(node[0], node[1], str(utility), zorder=3)

            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)

        if planned_paths:
            robot = self.robot_list[0]
            for i, path in enumerate(planned_paths):
                if path:
                    c = color_list[i]
                    plt.plot((np.array(path)[:, 0] - robot.map_info.map_origin_x) / robot.cell_size,
                            (np.array(path)[:, 1] - robot.map_info.map_origin_y) / robot.cell_size, c,
                            linewidth=2, zorder=1)

        if len(self.env.global_frontiers) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)), self.env.belief_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        # Agent Edges
        # for coords in self.robot_list[0].node_coords:
        #     node = self.robot_list[0].node_manager.nodes_dict.find(coords.tolist()).data
        #     for neighbor_coords in node.neighbor_list[1:]:
        #         end = (np.array(neighbor_coords) - coords) / 2 + coords
        #         plt.plot((np.array([coords[0], end[0]]) - self.robot_list[0].map_info.map_origin_x) / self.robot_list[0].cell_size,
        #                        (np.array([coords[1], end[1]]) - self.robot_list[0].map_info.map_origin_y) / self.robot_list[0].cell_size, 'tan', zorder=1)

        # # Ground Truth Edges
        # for coords in self.env.ground_truth_planner.ground_truth_node_manager.ground_truth_node_coords:
        #     node = self.env.ground_truth_planner.ground_truth_node_manager.ground_truth_nodes_dict.find(coords.tolist()).data
        #     for neighbor_coords in node.neighbor_list[1:]:
        #         end = (np.array(neighbor_coords) - coords) / 2 + coords
        #         plt.plot((np.array([coords[0], end[0]]) - self.env.ground_truth_info.map_origin_x) / self.env.cell_size,
        #                        (np.array([coords[1], end[1]]) - self.env.ground_truth_info.map_origin_y) / self.env.cell_size, 'tan', zorder=1)
        
        plt.axis('off')
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.env.explored_rate,
                                                                              max([robot.travel_dist for robot in
                                                                                   self.robot_list])))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()
