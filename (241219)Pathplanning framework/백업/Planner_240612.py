from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgGoal import MsgRequestManeuver
from Models.Message.MsgControl import MsgRequestManeuverControl
from Models.Message.MsgStopSimulation import MsgStopSimulation
from Models.Message.MsgDone import *
import math
import re
from enum import Enum
from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

def preprocess_terrain(terrain_polygons, buffer=-0.5):
    # 모든 지형 폴리곤을 합친 후, 지정된 버퍼만큼 축소
    merged_polygons = unary_union([Polygon(poly) for poly in terrain_polygons])
    accessible_area = merged_polygons.buffer(buffer)
    return accessible_area

def plot_path_with_terrain(start, goal, path, terrain_polygons):
    fig, ax = plt.subplots()
    
    # Plot terrain polygons
    for polygon in terrain_polygons:
        poly_patch = pltPolygon(polygon, closed=True, color='green', alpha=0.5)
        ax.add_patch(poly_patch)
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', label='Start')
    ax.plot(goal[0], goal[1], 'ro', label='Goal')
    
    # Plot path
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
    
    # Set axis limits and labels
    # ax.set_xlim(-10, 60)
    # ax.set_ylim(-10, 60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('A* Path with Terrain')
    ax.legend()
    
    plt.show()

def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    return ''.join(numbers)

class RobotType(Enum):
    circle = 0
    rectangle = 1

def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def line_of_sight(p1, p2, terrain_polygons):
    line = LineString([p1, p2])
    for polygon_points in terrain_polygons:
        polygon = Polygon(polygon_points)
        if line.intersects(polygon):
            return False
    return True

def astar(start, goal, terrain_polygons, grid_size=1):
    buffer_distance = 2  # Distance around the terrain polygons to increase the cost
    half_cost_factor = 0.5  # Factor to apply for the buffer cost

    # Create buffered polygons
    buffered_polygons = [Polygon(p).buffer(buffer_distance) for p in terrain_polygons]

    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return simplify_path(path, terrain_polygons)

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-grid_size, 0), (grid_size, 0), (0, -grid_size), (0, grid_size), 
                           (-grid_size, -grid_size), (-grid_size, grid_size), (grid_size, -grid_size), (grid_size, grid_size)]
        ]

        for neighbor in neighbors:
            neighbor_point = Point(neighbor)
            if any(Polygon(p).contains(neighbor_point) for p in terrain_polygons):
                continue  # Skip if neighbor is within any terrain polygon

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            # Increase cost if within buffer
            if any(buffered.contains(neighbor_point) for buffered in buffered_polygons):
                tentative_g_score += half_cost_factor * heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return []


# def simplify_path(path, terrain_polygons):
#     if len(path) < 3:
#         return path
#     simplified_path = [path[0]]
#     for i in range(2, len(path)):
#         if not line_of_sight(simplified_path[-1], path[i], terrain_polygons):
#             simplified_path.append(path[i - 1])
#     simplified_path.append(path[-1])
#     return simplified_path

def simplify_path(path, tolerance=0.1):
    line = LineString(path)
    simplified_line = line.simplify(tolerance, preserve_topology=True)
    return np.array(simplified_line.coords)

class Planner(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objLogger = None
        self.ID = ID

        self.objConfiguration = objConfiguration

        # 설정 값을 가져와서 속성으로 저장
        self.robot_radius = objConfiguration.getConfiguration('robot_radius')
        self.max_speed = objConfiguration.getConfiguration('max_speed')
        self.min_speed = objConfiguration.getConfiguration('min_speed')
        self.max_yaw_rate = objConfiguration.getConfiguration('max_yaw_rate')
        self.max_accel = objConfiguration.getConfiguration('max_accel')
        self.max_delta_yaw_rate = objConfiguration.getConfiguration('max_delta_yaw_rate')
        self.v_resolution = objConfiguration.getConfiguration('v_resolution')
        self.yaw_rate_resolution = objConfiguration.getConfiguration('yaw_rate_resolution')
        self.dt = objConfiguration.getConfiguration('dt')
        self.predict_time = objConfiguration.getConfiguration('predict_time')
        self.to_goal_cost_gain = objConfiguration.getConfiguration('to_goal_cost_gain')
        self.speed_cost_gain = objConfiguration.getConfiguration('speed_cost_gain')
        self.obstacle_cost_gain = objConfiguration.getConfiguration('obstacle_cost_gain')
        self.heuristic_cost_gain = objConfiguration.getConfiguration('heuristic_cost_gain')
        self.robot_type = objConfiguration.getConfiguration('robot_type')
        self.robot_stuck_flag_cons = objConfiguration.getConfiguration('robot_stuck_flag_cons')
        self.terrain_polygons= objConfiguration.getConfiguration('terrain_polygons')

        self.obstacles = {}  # 장애물 데이터를 저장할 딕셔너리
        self.other_agents = {}  # 다른 에이전트 데이터를 저장할 딕셔너리
        self.ob = np.array([])
        self.id = int(extract_numbers(self.ID))
        self.starts = objConfiguration.getConfiguration('agent_starts')
        self.goals = objConfiguration.getConfiguration('agent_goals')
        self.curPose = self.starts[self.id]
        # self.goal = goals[self.id]
        self.goal = None
        
        self.target_tolerance = 0.5 * self.robot_radius
        lstTeamComposition = [[ID]]
        # self.objAIPlanner = objAIPlanner

        self.addInputPort("MyManeuverState")
        self.addInputPort("OtherManeuverState")
        self.addInputPort("StopSimulation_IN")
        self.addInputPort("Goal_IN")
        self.addOutputPort('Done_OUT')

        for i in range(1):
            for j in range(1):
                strName = lstTeamComposition[i][j]
                self.addOutputPort(strName+"_RequestManeuver")
                self.addOutputPort(strName+"_GunFire")
                self.addOutputPort(strName+"_StopSimulation")

        self.addStateVariable("mode", 'WAIT')
        self.addStateVariable('robot_radius', self.robot_radius)
        self.addStateVariable('max_speed', self.max_speed)
        self.addStateVariable('min_speed', self.min_speed)
        self.addStateVariable('max_yaw_rate', self.max_yaw_rate)
        self.addStateVariable('max_accel', self.max_accel)
        self.addStateVariable('max_delta_yaw_rate', self.max_delta_yaw_rate)
        self.addStateVariable('v_resolution', self.v_resolution)
        self.addStateVariable('yaw_rate_resolution', self.yaw_rate_resolution)
        self.addStateVariable('dt', self.dt)
        self.addStateVariable('predict_time', self.predict_time)
        self.addStateVariable('to_goal_cost_gain', self.to_goal_cost_gain)
        self.addStateVariable('speed_cost_gain', self.speed_cost_gain)
        self.addStateVariable('obstacle_cost_gain', self.obstacle_cost_gain)
        self.addStateVariable('heuristic_cost_gain', self.heuristic_cost_gain)
        self.addStateVariable('robot_type', self.robot_type)
        self.addStateVariable('robot_stuck_flag_cons', self.robot_stuck_flag_cons)
        # self.addStateVariable('goal', self.goal)
        self.addStateVariable("target_linearVelocity", 0)
        self.addStateVariable("target_angularVelocity", 0)
        self.addStateVariable("path_index", 0)  # 현재 경유지 인덱스

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "MyManeuverState":
            self.curPose[0]= objEvent.x
            self.curPose[1]= objEvent.y
            self.curPose[2]= objEvent.yaw
            self.curPose[3]= objEvent.lin_vel
            self.curPose[4]= objEvent.ang_vel

        elif strPort == "OtherManeuverState" and self.getStateValue("mode") != 'ARRIVE':

            if "Obstacle" in objEvent.strID:
                self.obstacles[objEvent.strID] = (objEvent.dblPositionN, objEvent.dblPositionE)
            else:
                if objEvent.strID[:8] != self.ID[:8]:  # 본인의 ID를 제외하고 저장
                    self.other_agents[objEvent.strID] = (objEvent.dblPositionN, objEvent.dblPositionE)
                    # 다른 에이전트의 위치를 장애물로 간주하여 추가
                    self.obstacles[objEvent.strID] = (objEvent.dblPositionN, objEvent.dblPositionE)

            self.ob = np.array(list(self.obstacles.values()))
            # self.setStateValue("mode", 'PLANNING')
        elif strPort == "Goal_IN" and self.getStateValue("mode") == "WAIT":
            # 목적지 설정 이벤트는 본인에게만 해당되어야 함
            if objEvent.strID == self.ID:
                self.goal = [objEvent.goal_x, objEvent.goal_y]
                # starts_x = int(self.curPose[0])
                # starts_y = int(self.curPose[1])
                
                # self.path = astar(tuple([starts_x, starts_y]), tuple(self.goal), self.terrain_polygons)
                # self.path = self.path[1:]
                # plot_path_with_terrain(self.starts[self.id], self.goal , self.path, self.terrain_polygons)
                self.setStateValue("mode", 'GPP')
                self.gpp_start_time = self.getTime()
                pass
        return True
    
    def funcOutput(self):
        if self.getStateValue('mode') == "LPP":
            objRequestMessage = MsgRequestManeuverControl(self.ID, 
                                                          self.getStateValue('target_linearVelocity'), 
                                                          self.getStateValue('target_angularVelocity'))
            self.addOutputEvent(self.ID+"_RequestManeuver", objRequestMessage)
        elif self.getStateValue('mode') == "ARRIVE":#도착했으니 정지해라
            objRequestMessage = MsgRequestManeuverControl(self.ID, 
                                                        0, 
                                                        0)
            self.addOutputEvent(self.ID+"_RequestManeuver", objRequestMessage)
            objRequestMessage = MsgDone(self.ID, 
                                        self.curPose[0], 
                                        self.curPose[1])
            self.addOutputEvent('Done_OUT', objRequestMessage)
        return True
    
    def calc_dynamic_window(self, x,  goal):
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt, x[3] + self.max_accel * self.dt, x[4] - self.max_delta_yaw_rate * self.dt, x[4] + self.max_delta_yaw_rate * self.dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw
    
    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x
    
    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory
            
    def calc_obstacle_cost(self, trajectory, ob):
        if ob is None or len(ob) == 0:
            return 0  # 장애물이 없거나 리스트가 비어있을 경우 비용은 0

        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0][:, np.newaxis] - ox[np.newaxis, :]
        dy = trajectory[:, 1][:, np.newaxis] - oy[np.newaxis, :]
        r = np.hypot(dx, dy)

        if self.robot_type == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= self.robot_length / 2
            right_check = local_ob[:, 1] <= self.robot_width / 2
            bottom_check = local_ob[:, 0] >= -self.robot_length / 2
            left_check = local_ob[:, 1] >= -self.robot_width / 2
            if (np.logical_and(np.logical_and(upper_check, right_check), np.logical_and(bottom_check, left_check))).any():
                return float("Inf")  # 장애물과 충돌하는 경우 무한 비용

        elif self.robot_type == RobotType.circle:
            if np.array(r <= self.robot_radius).any():
                return float("Inf")  # 원형 로봇이 장애물과 충돌하는 경우 무한 비용

        min_r = np.min(r)  # 장애물과의 최소 거리를 기반으로 비용 계산
        return 1.0 / min_r if min_r != 0 else float("Inf")  # 최소 거리가 0이면 무한 비용, 아니면 역수로 비용 계산

    def calc_terrain_cost(self, trajectory, terrain_polygons):
        cost = 0.0
        for polygon_points in terrain_polygons:
            polygon = Polygon(polygon_points)
            for point in trajectory[:, :2]:
                p = Point(point[0], point[1])
                if polygon.contains(p):
                    cost += float("Inf")
                    print(f'calc_terrain_cost: {cost}')
                else:
                    cost = 0
        return cost

    def calc_to_goal_cost(self, trajectory, goal):
        if not trajectory.any() or len(trajectory) == 0:
            raise ValueError("Trajectory is empty or undefined.")
        if not goal or len(goal) != 2:
            raise ValueError("Goal position must be a tuple or list of two elements (x, y).")
        
        # Calculate the direction to the goal
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)  # Direction from current position to goal

        # Calculate the difference angle between the agent's current heading and the goal direction
        current_heading = trajectory[-1, 2]
        cost_angle = error_angle - current_heading
        
        # Normalize the cost angle to be within the range of -pi to pi
        cost_angle = (cost_angle + math.pi) % (2 * math.pi) - math.pi

        # Calculate the absolute minimal angle difference
        cost = min(cost_angle, 2 * math.pi - abs(cost_angle))

        return abs(cost)

    def euclidean_distance(self, x, y):
        return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def calc_to_goal_dist(self, trajectory, goal):
        h = self.euclidean_distance([trajectory[-1, 0], trajectory[-1, 1]], [goal[0], goal[1]])
        return h

    def calc_control_and_trajectory(self, x, dw, goal, ob, terrain_polygons):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)
                heuristic = self.heuristic_cost_gain * self.calc_to_goal_dist(trajectory, goal)
                terrian_cost = self.calc_terrain_cost(trajectory, terrain_polygons)
                # print(f'Obstacle cost:{ob_cost}')
                final_cost = to_goal_cost + speed_cost + ob_cost + heuristic + terrian_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons and abs(x[3]) < self.robot_stuck_flag_cons:
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory

    def funcInternalTransition(self):
        if self.getStateValue('mode') == 'LPP':
            dist_to_goal = math.hypot(self.curPose[0] - self.goal[0], self.curPose[1] - self.goal[1])
            if dist_to_goal <= self.target_tolerance:
                print("Goal reached.")
                self.setStateValue("mode", 'ARRIVE')
                self.setStateValue("path_index", 0)
            else:
                path_index = self.getStateValue("path_index")

                if path_index < len(self.path):
                    next_goal = self.path[path_index]
                    dw = self.calc_dynamic_window(self.curPose, next_goal)
                    u, trajectory = self.calc_control_and_trajectory(self.curPose, dw, next_goal, self.ob, self.terrain_polygons)
                    self.setStateValue("target_linearVelocity", u[0])
                    self.setStateValue("target_angularVelocity", u[1])
                    if np.linalg.norm(np.array(self.curPose[:2]) - np.array(next_goal)) <= 0.5:
                        self.setStateValue("path_index", path_index + 1)
                elif path_index == len(self.path)-1:
                    next_goal = self.path[path_index]
                    dw = self.calc_dynamic_window(self.curPose, next_goal)
                    u, trajectory = self.calc_control_and_trajectory(self.curPose, dw, next_goal, self.ob, self.terrain_polygons)
                    self.setStateValue("target_linearVelocity", u[0])
                    self.setStateValue("target_angularVelocity", u[1])
                    if np.linalg.norm(np.array(self.curPose[:2]) - np.array(next_goal)) <= self.target_tolerance:
                        self.setStateValue("path_index", path_index + 1)
                else:
                    # self.setStateValue("mode", 'WAIT')
                    self.setStateValue("path_index", 0)
            current_time= self.getTime()   

            if current_time%1000 < 0.1:
                self.setStateValue('mode', 'GPP')
                
 
        elif self.getStateValue('mode') == 'ARRIVE':
            self.setStateValue("mode","WAIT")

        elif self.getStateValue('mode') == 'GPP':
            starts_x = int(self.curPose[0])
            starts_y = int(self.curPose[1])
            
            self.path = astar(tuple([starts_x, starts_y]), tuple(self.goal), self.terrain_polygons)
            self.path = self.path[1:]
            self.setStateValue("path_index", 0)
            # plot_path_with_terrain(self.starts[self.id], self.goal , self.path, self.terrain_polygons)
            self.setStateValue("mode","LPP")
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "LPP":
            return 0.1
        elif self.getStateValue('mode') == "GPP" or self.getStateValue('mode') == "ARRIVE":
            return 0.0
        else:
            return 999999999
    
    def funcSelect(self):
        pass
