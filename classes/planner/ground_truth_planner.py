from time import time
from copy import deepcopy

import numpy as np
from sympy import vring

from parameter import *
from classes.utils import *
from classes.planner.ortools_solver import solve_vrp
import classes.agent.quads as quads

np.random.seed(47)


class GroundTruthPlanner:
    def __init__(self, ground_truth_info, node_manager):
        self.max_iteration_step = 10 # you could increase this to get better results at the cost of more computation
        self.ground_truth_info = deepcopy(ground_truth_info)
        self.ground_truth_node_manager = GroundTruthNodeManager(self.ground_truth_info, node_manager)
        self.last_viewpoints = None

    def plan_coverage_paths(self, belief_info, robot_locations, map_change):
        self.ground_truth_node_manager.update_ground_truth_graph(belief_info)
        for coords in robot_locations:
            node = self.ground_truth_node_manager.ground_truth_nodes_dict.find(coords.tolist())
            node.data.set_visited()

        best_paths = None
        c_best = 1e10
        q_indices = np.where(self.ground_truth_node_manager.utility > 0)[0]
        q_array = self.ground_truth_node_manager.ground_truth_node_coords[q_indices]

        node = self.ground_truth_node_manager.ground_truth_nodes_dict.find(robot_locations[0].tolist()).data
        dist_dict = node.dist_dict

        if self.last_viewpoints:
            temp_node_manager = TempNodeManager()
            for node in self.ground_truth_node_manager.ground_truth_nodes_dict.__iter__():
                coords = node.data.coords
                observable_frontiers = deepcopy(node.data.observable_frontiers)
                temp_node_manager.add_node_to_dict(coords, observable_frontiers)
            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]

            for viewpoints in self.last_viewpoints:
                last_node = temp_node_manager.nodes_dict.find(viewpoints.tolist()).data
                if last_node.utility > 0:
                    v_list.append(last_node.coords)
                    observable_frontiers = last_node.observable_frontiers
                    index = np.argwhere(last_node.coords[0] + last_node.coords[1] * 1j == q_array_prime[:, 0] + q_array_prime[:, 1] * 1j)[0][0]
                    q_array_prime = np.delete(q_array_prime, index, axis=0)
                    for coords in q_array_prime:
                        node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                        if node.utility > 0 and np.linalg.norm(coords - last_node.coords) < 2 * SENSOR_RANGE:
                            node.delete_observed_frontiers(observable_frontiers)

            q_utility = []
            for coords in q_array_prime:
                node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                q_utility.append(node.utility)
            q_utility = np.array(q_utility)

            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0:
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                # assert viewpoint_coords[0] + viewpoint_coords[1] * 1j not in v_list[:][0] + v_list[:][1] * 1j
                node = temp_node_manager.nodes_dict.find(viewpoint_coords.tolist()).data
                if dist_dict[(node.coords[0], node.coords[1])] == 1e8:
                    observable_frontiers = np.array([[], []]).reshape(-1, 2)
                else:
                    v_list.append(viewpoint_coords)
                    observable_frontiers = np.array(node.observable_frontiers)
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                    if node.utility > 0 and np.linalg.norm(coords - viewpoint_coords) < 2 * SENSOR_RANGE:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations)
            best_paths = paths
            c_best = dist

            if not map_change:
                self.last_viewpoints = v_list[len(robot_locations):]
                return best_paths

        for i in range(self.max_iteration_step):
            temp_node_manager = TempNodeManager()
            for node in self.ground_truth_node_manager.ground_truth_nodes_dict.__iter__():
                coords = node.data.coords
                observable_frontiers = deepcopy(node.data.observable_frontiers)
                temp_node_manager.add_node_to_dict(coords, observable_frontiers)

            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]
            q_utility = self.ground_truth_node_manager.utility[q_indices]

            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0:
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                # assert viewpoint_coords[0] + viewpoint_coords[1] * 1j not in v_list[:][0] + v_list[:][1] * 1j
                node = temp_node_manager.nodes_dict.find(viewpoint_coords.tolist()).data
                if dist_dict[(node.coords[0], node.coords[1])] == 1e8:
                    observable_frontiers = np.array([[], []]).reshape(-1, 2)
                else:
                    v_list.append(viewpoint_coords)
                    observable_frontiers = node.observable_frontiers
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                    if node.utility > 0:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations)
            t3 = time()
            if dist < c_best:
                best_paths = paths
                c_best = dist
                self.last_viewpoints = v_list[len(robot_locations):]

        return best_paths

    def find_paths(self, viewpoints, robot_locations):
        t1 = time()
        size = len(viewpoints)
        path_matrix = []
        distance_matrix = np.ones((size, size), dtype=int) * 1000
        for i in range(size):
            path_matrix.append([])
            for j in range(size):
                path_matrix[i].append([])

        for i in range(size):
            node = self.ground_truth_node_manager.ground_truth_nodes_dict.find(viewpoints[i].tolist()).data
            for j in range(size):
                path, dist = node.get_path_and_dist_to_node(viewpoints[j])
                dist = dist.astype(int)

                distance_matrix[i][j] = dist
                path_matrix[i][j] = path

        robot_indices = [i for i in range(len(robot_locations))]
        for i in range(size):
            for j in robot_indices:
                distance_matrix[i][j] = 0

        paths, max_dist = solve_vrp(distance_matrix, robot_indices)
        t3 = time()

        paths_coords = []
        for path, robot_location in zip(paths, robot_locations):
            path_coords = []
            for index1, index2 in zip(path[:-1], path[1:]):
                path_coords += path_matrix[index1][index2]
            if len(path_coords) == 0:
                indices = np.argwhere(self.ground_truth_node_manager.utility > 0).reshape(-1)
                node_coords = self.ground_truth_node_manager.ground_truth_node_coords[indices]
                node = self.ground_truth_node_manager.ground_truth_nodes_dict.find(robot_location.tolist()).data
                dist_dict, prev_dict = node.dist_dict, node.path_dict
                nearest_utility_coords = robot_location
                nearest_dist = 1e8
                for coords in node_coords:
                    dist = dist_dict[(coords[0], coords[1])]
                    if 0 < dist < nearest_dist:
                        nearest_dist = dist
                        nearest_utility_coords = coords
                    if nearest_dist == 1e8 and dist != 0:
                        print(dist, robot_location, coords)
                        # print(nearest_dist, coords, nearest_utility_coords, robot_location)

                path_coords, dist = self.ground_truth_node_manager.a_star(robot_location, nearest_utility_coords)
                if len(path_coords) == 0:
                    print("nearest", nearest_utility_coords, robot_location, node_coords.shape, nearest_dist)

            paths_coords.append(path_coords)
        return paths_coords, max_dist


class GroundTruthNodeManager:
    def __init__(self, ground_truth_map_info, node_manager):
        self.ground_truth_nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.ground_truth_map_info = ground_truth_map_info
        self.ground_truth_map_info.map = ((self.ground_truth_map_info.map == FREE) * (FREE - UNKNOWN)) + UNKNOWN
        self.ground_truth_frontiers = get_frontier_in_map(self.ground_truth_map_info, voxel_size=CELL_SIZE)
        self.ground_truth_node_coords, self.utility = self.initial_ground_truth_graph()
        self.belief_node_manager = node_manager

    def get_ground_truth_node_coords(self):
        x_min = self.ground_truth_map_info.map_origin_x
        y_min = self.ground_truth_map_info.map_origin_y
        x_max = self.ground_truth_map_info.map_origin_x + (self.ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = self.ground_truth_map_info.map_origin_y + (self.ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1 , NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, self.ground_truth_map_info)
        for i, cell in enumerate(nodes_cells):
            if self.ground_truth_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices]

        return nodes

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = GroundTruthNode(coords, self.ground_truth_frontiers, self.ground_truth_map_info)
        self.ground_truth_nodes_dict.insert(point=key, data=node)

    def initial_ground_truth_graph(self):
        ground_truth_node_coords = self.get_ground_truth_node_coords()
        for coords in ground_truth_node_coords:
            self.add_node_to_dict(coords)
        for coords in ground_truth_node_coords:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            node.update_neighbor_nodes(self.ground_truth_map_info, self.ground_truth_nodes_dict)
        for node in self.ground_truth_nodes_dict.__iter__():
            node = node.data
            node.dist_dict, node.path_dict = self.Dijkstra(node.coords)

        utility = []
        for coords in ground_truth_node_coords:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
        utility = np.array(utility)

        return ground_truth_node_coords, utility

    def update_ground_truth_graph(self, belief_map_info):
        observed_obstacles_map = belief_map_info.map
        updated_map = self.ground_truth_map_info.map
        updated_map = np.where(observed_obstacles_map == OCCUPIED, observed_obstacles_map, updated_map)
        self.ground_truth_map_info.map = updated_map

        self.ground_truth_frontiers = get_frontier_in_map(self.ground_truth_map_info, voxel_size=CELL_SIZE)

        if len(self.ground_truth_frontiers) > 0:
            observed_frontiers = []
            frontier_in_cell = get_cell_position_from_coords(np.array(list(self.ground_truth_frontiers)), belief_map_info).reshape(-1, 2)
            for frontier, cell in zip(self.ground_truth_frontiers, frontier_in_cell):
                if belief_map_info.map[cell[1], cell[0]] == FREE:
                    observed_frontiers.append(frontier)
            for frontier in observed_frontiers:
                self.ground_truth_frontiers.remove(frontier)

        for node in self.ground_truth_nodes_dict.__iter__():
            node_in_belief = self.belief_node_manager.nodes_dict.find(node.data.coords.tolist())
            if node_in_belief is not None:
                frontiers = node_in_belief.data.observable_frontiers
            else:
                frontiers = set()
            if node.data.utility > 0:
                node.data.update_node_observable_frontiers(self.ground_truth_frontiers, frontiers, belief_map_info)

        utility = []
        for coords in self.ground_truth_node_coords:
            node = self.ground_truth_nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
        self.utility = np.array(utility)

    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.ground_truth_nodes_dict.find(key)
        return exist

    def a_star(self, start, destination, max_dist=1e8):
        if not self.check_node_exist_in_dict(start):
            print('start does not existed')
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            print('destination does not existed')
            return [], 1e8
        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.ground_truth_nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.ground_truth_nodes_dict.find(n).data
                    n_coords = node.coords

            # if g[n] > max_dist:
            #     return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if g[n] + cost > max_dist:
                    continue
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')

        return [], 1e8

    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.ground_truth_nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:

            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            node = self.ground_truth_nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)


class GroundTruthNode:
    def __init__(self, coords, ground_truth_frontiers, ground_truth_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.observable_frontiers = self.initialize_observable_frontiers(ground_truth_frontiers, ground_truth_map_info)
        self.utility = len(self.observable_frontiers)

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)

        self.dist_dict = None
        self.path_dict = None

    def initialize_observable_frontiers(self, ground_truth_frontiers, ground_truth_map_info):
        if len(ground_truth_frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            ground_truth_frontiers = np.array(list(ground_truth_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(ground_truth_frontiers - self.coords, axis=-1)
            frontiers_in_range = ground_truth_frontiers[dist_list < self.utility_range]
            for point in frontiers_in_range:
                collision = check_collision(self.coords, point, ground_truth_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            return observable_frontiers

    def update_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        # cell = get_cell_position_from_coords(neighbor_coords, ground_truth_map_info)
                        # if cell[0] < ground_truth_map_info.map.shape[1] and cell[1] < ground_truth_map_info.map.shape[0]:
                        #     if ground_truth_map_info.map[cell[1], cell[0]] == 1:
                        #         self.neighbor_matrix[i, j] = 1
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, ground_truth_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

    def update_node_observable_frontiers(self, ground_truth_frontiers, frontiers, belief_info):
        self.observable_frontiers = self.observable_frontiers & ground_truth_frontiers

        new_frontiers = frontiers - self.observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, belief_info)
            if not collision:
                self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.observable_frontiers = set()
            self.utility = 0

    def delete_observed_frontiers(self, observed_frontiers):
        # remove observed frontiers in the observable frontiers
        self.observable_frontiers = self.observable_frontiers - observed_frontiers

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.observable_frontiers = set()
            self.utility = 0

    def get_path_and_dist_to_node(self, coords):
        dist = self.dist_dict[(coords[0], coords[1])]

        path = [(coords[0], coords[1])]
        prev_node = self.path_dict[(coords[0], coords[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = self.path_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def set_visited(self):
        self.observable_frontiers = set()
        self.utility = 0


class TempNodeManager:
    def __init__(self):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)

    def add_node_to_dict(self, coords, observable_frontiers):
        key = (coords[0], coords[1])
        node = TempNode(coords, observable_frontiers)
        self.nodes_dict.insert(point=key, data=node)

class TempNode:
    def __init__(self, coords, observable_frontiers):
        self.coords = coords
        self.observable_frontiers = observable_frontiers
        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.observable_frontiers = set()
            self.utility = 0

    def delete_observed_frontiers(self, observed_frontiers):
        # remove observed frontiers in the observable frontiers
        self.observable_frontiers = self.observable_frontiers - observed_frontiers

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.observable_frontiers = set()
            self.utility = 0
