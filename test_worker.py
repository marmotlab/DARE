import os
import time
import random
import collections

from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt

from test_parameter import *
from classes.utils import *
from classes.env.env import Env
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager

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

class TestWorker:
    def __init__(self, meta_agent_id, policy, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.policy = policy
        self.global_step = global_step
        self.device = device
        self.save_image = save_image

        self.env= Env(global_step, TEST_N_AGENTS, plot=save_image, test=USE_TEST_DATASET)
        self.node_manager = NodeManager(plot=save_image)
        self.robot_list = [Agent(i, self.node_manager, self.device, save_image) for i in range(TEST_N_AGENTS)]
        self.perf_metrics = dict()

        self.obs_horizon = policy.n_obs_steps
        self.action_horizon = policy.n_action_steps if ACTION_HORIZON == None else ACTION_HORIZON
        # assert self.action_horizon == 1, "Action horizon must be 1 for this driver"

        self.planned_path_x = []
        self.planned_path_y = []

    def run_episode(self):
        unique_seed = int(time.time())
        set_random_seed(unique_seed)
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        # Get the first observation
        if DATA_TYPE == 'node':
            observation = self.robot_list[0].get_observation()
            node_inputs = observation[0].squeeze(0)
            node_padding_mask = observation[1].squeeze(0)
            edge_mask = observation[2].squeeze(0)
            current_index = observation[3].squeeze(0)
            current_edge = observation[4].squeeze(0)
            edge_padding_mask = observation[5].squeeze(0)

            obs = {'node_inputs': node_inputs,
                'node_padding_mask': node_padding_mask,
                'edge_mask': edge_mask,
                'current_index': current_index,
                'current_edge': current_edge,
                'edge_padding_mask': edge_padding_mask}
        elif DATA_TYPE == 'map':
            image = deepcopy(self.env.robot_belief)
            state = deepcopy(self.env.robot_locations[0])

            agent_pos = state.astype(np.float32)
            image = image.astype(np.float32)/255
            image = np.expand_dims(image, axis=0)

            obs = {'image': image,
                'agent_pos': state}
        else:
            raise ValueError('Invalid data type, check test_parameter.py')

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        step = 0
        for step in range(MAX_EPISODE_STEP):
            # stack the last obs_horizon number of observations
            if DATA_TYPE == 'node':
                node_inputs = torch.stack([x['node_inputs'] for x in obs_deque])
                node_padding_mask = torch.stack([x['node_padding_mask'] for x in obs_deque])
                edge_mask = torch.stack([x['edge_mask'] for x in obs_deque])
                current_index = torch.stack([x['current_index'] for x in obs_deque])
                current_edge = torch.stack([x['current_edge'] for x in obs_deque])
                edge_padding_mask = torch.stack([x['edge_padding_mask'] for x in obs_deque])

                # device transfer
                # TODO check if need to down cast to int16/32 then upcast to int64 like dataset
                node_inputs = node_inputs.to(self.device, dtype=torch.float32) # (obs_horizon, 360, 5)
                node_padding_mask = node_padding_mask.to(self.device, dtype=torch.int16) # (obs_horizon, 1, 360)
                edge_mask = edge_mask.to(self.device, dtype=torch.int64) # (obs_horizon, 360, 360)
                current_index = current_index.to(self.device, dtype=torch.int64) # (obs_horizon, 1, 1)
                current_edge = current_edge.to(self.device, dtype=torch.int64) # (obs_horizon, 25, 1)
                edge_padding_mask = edge_padding_mask.to(self.device, dtype=torch.int16) # (obs_horizon, 1, 25)

                # observation dict
                obs_dict = {'node_inputs': node_inputs.unsqueeze(0),
                            'node_padding_mask': node_padding_mask.unsqueeze(0),
                            'edge_mask': edge_mask.unsqueeze(0),
                            'current_index': current_index.unsqueeze(0),
                            'current_edge': current_edge.unsqueeze(0),
                            'edge_padding_mask': edge_padding_mask.unsqueeze(0)}
            elif DATA_TYPE == 'map':
                # stack the last obs_horizon number of observations
                image = torch.stack([torch.tensor(x['image']) for x in obs_deque])
                agent_pos = torch.stack([torch.tensor(x['agent_pos']) for x in obs_deque])

                # device transfer
                image = image.to(self.device, dtype=torch.float32) # (obs_horizon, 512)
                agent_pos = agent_pos.to(self.device, dtype=torch.float32) # (obs_horizon, 2)

                # observation dict
                obs_dict = {'image': image.unsqueeze(0),
                            'agent_pos': agent_pos.unsqueeze(0)}
            else:
                raise ValueError('Invalid data type, check test_parameter.py')
        
            # infer action
            # time_start = time.time()
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)
            # time_end = time.time()
            # print(f"Time taken for inference: {time_end - time_start}s")

            action_pred = action_dict['action_pred'].squeeze(0).cpu().numpy() # (pred_horizon, action_dim)

            action_pred = np.round(action_pred / NODE_RESOLUTION) * NODE_RESOLUTION  # round to nearest node resolution
            
            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            action = action_pred[start:end,:] # (action_horizon, action_dim)
            
            # execute action_horizon number of steps without replanning
            for action_step in range(self.action_horizon):
                if action_step == 0:
                    planned_location = deepcopy(self.env.robot_locations[0])
                    self.planned_path_x.append([planned_location[0]])
                    self.planned_path_y.append([planned_location[1]])
                    # get planned path for visualization
                    if USE_DELTA_POSITION:
                        for i in range(start, len(action_pred)):
                            planned_location = planned_location + action_pred[i]
                            self.planned_path_x[step].append(planned_location[0])
                            self.planned_path_y[step].append(planned_location[1])
                    else:
                        for i in range(start, len(action_pred)):
                            planned_location = action_pred[i]
                            self.planned_path_x[step].append(planned_location[0])
                            self.planned_path_y[step].append(planned_location[1])
                else:
                    self.planned_path_x.append(self.planned_path_x[step - action_step])
                    self.planned_path_y.append(self.planned_path_y[step - action_step])
                    pass
                # print(f"Step: {step}, Action Step: {action_step}")
                if USE_DELTA_POSITION:
                    selected_coord = self.env.robot_locations[0] + action[action_step]
                else:
                    selected_coord = action[action_step]

                current_node = self.robot_list[0].node_manager.nodes_dict.find(self.env.robot_locations[0].tolist()).data

                ## Collision avoidance
                # check if selected_coord is a valid neighbour of current node
                if not any(np.all(selected_coord == neighbor) for neighbor in current_node.neighbor_list):
                    # print("Collision Detected!")
                    # Vectors of 3 future positions from current position # HACK fixed number here
                    direction_vectors = np.cumsum(action_pred[start: start + 3], axis=0)
                    best_neighbor = None
                    best_average_angle = float('inf')
                    # print(f"Direction Vectors: {direction_vectors}")
                    for neighbor_coords in current_node.neighbor_list:
                        # skip current robot location
                        if np.all(neighbor_coords == self.env.robot_locations[0]):
                            continue 
                        neighbor_direction = neighbor_coords - self.env.robot_locations[0]
                        # print(f"Neighbor Direction: {neighbor_direction}")
                        angles = []
                        for direction_vector in direction_vectors:
                            direction_magnitude = np.linalg.norm(direction_vector)
                            neighbor_magnitude = np.linalg.norm(neighbor_direction)
                            if direction_magnitude == 0 or neighbor_magnitude == 0: # skip zero vectors
                                continue
                            angle = np.arctan2(np.linalg.det([direction_vector, neighbor_direction]), np.dot(direction_vector, neighbor_direction))
                            angles.append(angle)
                        weights = np.arange(len(angles), 0, -1)
                        weighted_average_angle = np.average(np.abs(angles), weights=weights)  # Use absolute values for magnitude
                        # print(f"Weighted Average Angle: {weighted_average_angle}")
                        if weighted_average_angle < best_average_angle:
                            best_average_angle = weighted_average_angle
                            best_neighbor = neighbor_coords
                    # print(f"Best Neighbor: {best_neighbor}, action: {best_neighbor - self.env.robot_locations[0]}")
                    selected_coord = best_neighbor
                else:
                    # print("Valid Action")
                    pass
                
                # step the environment
                self.env.step(selected_coord, 0)
                
                # update robot state
                self.robot_list[0].update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[0]))
                self.robot_list[0].update_planning_state(self.env.robot_locations)

                if DATA_TYPE == 'node':
                    observation = self.robot_list[0].get_observation()
                    node_inputs = observation[0].squeeze(0)
                    node_padding_mask = observation[1].squeeze(0)
                    edge_mask = observation[2].squeeze(0)
                    current_index = observation[3].squeeze(0)
                    current_edge = observation[4].squeeze(0)
                    edge_padding_mask = observation[5].squeeze(0)

                    obs = {'node_inputs': node_inputs,
                        'node_padding_mask': node_padding_mask,
                        'edge_mask': edge_mask,
                        'current_index': current_index,
                        'current_edge': current_edge,
                        'edge_padding_mask': edge_padding_mask}
                elif DATA_TYPE == 'map':
                    image = deepcopy(self.env.robot_belief)
                    state = deepcopy(self.env.robot_locations[0])

                    agent_pos = state.astype(np.float32)
                    image = image.astype(np.float32)/255
                    if len(image.shape) == 2: # add channel dimension
                        image = np.expand_dims(image, axis=0)

                    obs = {'image': image,
                        'agent_pos': state}
                else:
                    raise ValueError('Invalid data type, check test_parameter.py')

                obs_deque.append(obs)

                if USE_EXPLORATION_RATE_FOR_DONE:
                    self.env.check_done()
                    done = self.env.done
                else:
                    done = self.robot_list[0].utility.sum() == 0

                if self.save_image: # save gif
                    self.plot_env(step)

                if done: # exit action loop if done or collision
                    break

            if done: # exit episode loop if done
                break
                
        self.perf_metrics['travel_dist'] = self.robot_list[0].travel_dist
        self.perf_metrics['success_rate'] = done
        
        if self.save_image: # save gif
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, delete_images=True)

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
            plt.plot((np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size,
                    (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size, 'g',
                    linewidth=1, zorder=2)

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

        if len(self.env.global_frontiers) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)), self.env.belief_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)

        # # Agent Edges
        # for coords in self.robot_list[0].node_coords:
        #     node = self.robot_list[0].node_manager.nodes_dict.find(coords.tolist()).data
        #     for neighbor_coords in node.neighbor_list[1:]:
        #         end = (np.array(neighbor_coords) - coords) / 2 + coords
        #         plt.plot((np.array([coords[0], end[0]]) - self.robot_list[0].map_info.map_origin_x) / self.robot_list[0].cell_size,
        #                         (np.array([coords[1], end[1]]) - self.robot_list[0].map_info.map_origin_y) / self.robot_list[0].cell_size, 'tan', zorder=1)

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