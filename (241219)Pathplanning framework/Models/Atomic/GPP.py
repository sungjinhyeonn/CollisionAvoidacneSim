import numpy as np
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon
from shapely.geometry import Point, LineString, Polygon

def RRT(start, goal, terrain_polygons, max_iterations=5000, step_size=1):
    class Node:
        def __init__(self, point, parent=None):
            self.point = point
            self.parent = parent

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def nearest_node(nodes, random_point):
        points = np.array([node.point for node in nodes])
        random_point = np.array(random_point)
        distances = np.linalg.norm(points - random_point, axis=1)
        return nodes[np.argmin(distances)]

    def steer(from_node, to_point, step_size):
        if distance(from_node.point, to_point) < step_size:
            return to_point
        else:
            from_point = np.array(from_node.point)
            to_point = np.array(to_point)
            direction = (to_point - from_point) / np.linalg.norm(to_point - from_point)
            new_point = from_point + step_size * direction
            return tuple(new_point)

    def line_collision_free(from_point, to_point, polygons):
        line = LineString([from_point, to_point])
        for poly in polygons:
            if poly.intersects(line):
                return False
        return True

    def generate_random_point(x_bounds, y_bounds):
        if np.random.random() < 0.05:
            return goal
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0], y_bounds[1])
        return (x, y)

    x_bounds = [min(start[0], goal[0]) - 10, max(start[0], goal[0]) + 10]
    y_bounds = [min(start[1], goal[1]) - 10, max(start[1], goal[1]) + 10]
    root = Node(start)
    nodes = [root]
    polygons = [Polygon(p) for p in terrain_polygons]

    for _ in range(max_iterations):
        random_point = generate_random_point(x_bounds, y_bounds)
        nearest = nearest_node(nodes, random_point)
        new_point = steer(nearest, random_point, step_size)
        if line_collision_free(nearest.point, new_point, polygons):
            new_node = Node(new_point, nearest)
            nodes.append(new_node)
            if distance(new_point, goal) <= step_size:
                return reconstruct_path(new_node)

    return []  # Return empty if no path is found

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.point)
        node = node.parent
    path.reverse()
    return path
