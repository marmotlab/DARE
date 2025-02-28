from copy import deepcopy

from classes.utils import *
from classes.planner.ortools_solver import solve_vrp
import classes.agent.quads as quads

class ExpertPlanner:
    def __init__(self, node_manager):
        self.max_iteration_step = 5
        self.node_manager = node_manager
        self.last_viewpoints = None

    def plan_coverage_paths(self, robot_locations):
        all_node_coords = []
        utility = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
            utility.append(node.data.utility)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = np.array(utility)
        
        best_paths = None
        c_best = 1e10
        q_indices = np.where(utility > 0)[0]
        q_array = all_node_coords[q_indices]

        dist_dict, _ = self.node_manager.Dijkstra(robot_locations[0])

        if self.last_viewpoints:
            temp_node_manager = TempNodeManager()
            for node in self.node_manager.nodes_dict.__iter__():
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
                    index = np.argwhere(
                        last_node.coords[0] + last_node.coords[1] * 1j == q_array_prime[:, 0] + q_array_prime[:,
                                                                                                1] * 1j)[0][0]
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
                    observable_frontiers = set()
                else:
                    v_list.append(viewpoint_coords)
                    observable_frontiers = node.observable_frontiers
                q_array_prime = np.delete(q_array_prime, sample, axis=0)
                q_utility = []
                for coords in q_array_prime:
                    node = temp_node_manager.nodes_dict.find(coords.tolist()).data
                    if node.utility > 0 and np.linalg.norm(coords - viewpoint_coords) <= 2 * SENSOR_RANGE:
                        node.delete_observed_frontiers(observable_frontiers)
                    q_utility.append(node.utility)
                q_utility = np.array(q_utility)

            paths, dist = self.find_paths(v_list, robot_locations, all_node_coords, utility)
            best_paths = paths
            c_best = dist

        for i in range(self.max_iteration_step):
            temp_node_manager = TempNodeManager()
            for node in self.node_manager.nodes_dict.__iter__():
                coords = node.data.coords
                observable_frontiers = deepcopy(node.data.observable_frontiers)
                temp_node_manager.add_node_to_dict(coords, observable_frontiers)

            q_array_prime = deepcopy(q_array)
            v_list = [location for location in robot_locations]
            q_utility = utility[q_indices]
            while q_array_prime.shape[0] > 0 and q_utility.sum() > 0:
                indices = np.array(range(q_array_prime.shape[0]))
                weights = q_utility / q_utility.sum()
                sample = np.random.choice(indices, size=1, replace=False, p=weights)[0]
                viewpoint_coords = q_array_prime[sample]
                # assert viewpoint_coords[0] + viewpoint_coords[1] * 1j not in v_list[:][0] + v_list[:][1] * 1j
                node = temp_node_manager.nodes_dict.find(viewpoint_coords.tolist()).data
                if dist_dict[(node.coords[0], node.coords[1])] == 1e8:
                    observable_frontiers = set()
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

            paths, dist = self.find_paths(v_list, robot_locations, all_node_coords, utility)
            if dist < c_best:
                best_paths = paths
                c_best = dist
                self.last_viewpoints = v_list[len(robot_locations):]

        return best_paths

    def find_paths(self, viewpoints, robot_locations, all_node_coords, utility):
        size = len(viewpoints)
        path_matrix = []
        distance_matrix = np.ones((size, size), dtype=int) * 1000
        for i in range(size):
            path_matrix.append([])
            for j in range(size):
                path_matrix[i].append([])

        for i in range(size):
            dist_dict, prev_dict = self.node_manager.Dijkstra(viewpoints[i])
            for j in range(size):
                path, dist = self.node_manager.get_Dijkstra_path_and_dist(dist_dict, prev_dict,
                                                                                       viewpoints[j])
                assert dist != 1e8

                dist = dist.astype(int)
                distance_matrix[i][j] = dist
                path_matrix[i][j] = path

        robot_indices = [i for i in range(len(robot_locations))]
        for i in range(size):
            for j in robot_indices:
                distance_matrix[i][j] = 0

        paths, max_dist = solve_vrp(distance_matrix, robot_indices)

        paths_coords = []
        for path, robot_location in zip(paths, robot_locations):
            path_coords = []
            for index1, index2 in zip(path[:-1], path[1:]):
                path_coords += path_matrix[index1][index2]
            if len(path_coords) == 0:
                indices = np.argwhere(utility > 0).reshape(-1)
                node_coords = all_node_coords[indices]
                dist_dict, prev_dict = self.node_manager.Dijkstra(robot_location)
                nearest_utility_coords = robot_location
                nearest_dist = 1e8
                for coords in node_coords:
                    dist = dist_dict[(coords[0], coords[1])]
                    if 0 < dist < nearest_dist:
                        nearest_dist = dist
                        nearest_utility_coords = coords

                path_coords, dist = self.node_manager.a_star(robot_location, nearest_utility_coords)
                if len(path_coords) == 0:
                    print("nearest", nearest_utility_coords, robot_location, node_coords.shape, nearest_dist)

            paths_coords.append(path_coords)
        return paths_coords, max_dist

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
