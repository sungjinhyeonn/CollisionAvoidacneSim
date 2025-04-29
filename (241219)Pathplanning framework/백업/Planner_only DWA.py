from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgGoal import MsgRequestManeuver
from Models.Message.MsgControl import MsgRequestManeuverControl
from Models.Message.MsgStopSimulation import MsgStopSimulation
import numpy as np
import math
from enum import Enum
import re
from shapely.geometry import Polygon, Point

def extract_numbers(input_string):
    # 정규 표현식을 사용하여 문자열에서 숫자만 추출
    numbers = re.findall(r'\d+', input_string)
    # 추출된 숫자를 하나의 문자열로 연결하여 반환
    return ''.join(numbers)

class RobotType(Enum):
    circle = 0
    rectangle = 1

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
        starts = objConfiguration.getConfiguration('agent_starts')
        goals = objConfiguration.getConfiguration('agent_goals')
        self.curPose = starts[self.id]
        self.goal = goals[self.id]
        self.dist_to_goal = 99999999
        self.target_tolerance = 1

        lstTeamComposition = [[ID]]
        # self.objAIPlanner = objAIPlanner

        self.addInputPort("MyManeuverState")
        self.addInputPort("OtherManeuverState")

        self.addInputPort("StopSimulation_IN")


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
        self.addStateVariable('goal', self.goal)
        self.addStateVariable("target_linearVelocity", 0)
        self.addStateVariable("target_angularVelocity", 0) 
        

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
            self.setStateValue("mode", 'PLANNING')
        return True
    
    def funcOutput(self):
        if self.getStateValue('mode') == "PLANNING":
            objRequestMessage = MsgRequestManeuverControl(self.ID, 
                                                          self.getStateValue('target_linearVelocity'), 
                                                          self.getStateValue('target_angularVelocity'))
            self.addOutputEvent(self.ID+"_RequestManeuver", objRequestMessage)
        elif self.getStateValue('mode') == "ARRIVE":
            objRequestMessage = MsgRequestManeuverControl(self.ID, 
                                                          0, 
                                                          0)
            self.addOutputEvent(self.ID+"_RequestManeuver", objRequestMessage)
            # objStopSimulation = MsgStopSimulation(self.ID)
            # if self.dist_to_goal< self.target_tolerance:
            #     self.addOutputEvent(self.ID+"_StopSimulation", objStopSimulation)
            #     self.setStateValue("mode", 'STOP')
        return True
    
        
    def calc_dynamic_window(self, x,  goal):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
            -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
            x[3] + self.max_accel * self.dt,
            x[4] - self.max_delta_yaw_rate * self.dt,
            x[4] + self.max_delta_yaw_rate * self.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw
    
    def motion(self, x, u, dt):
        """
        motion model
        """

        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x
    
    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory
            
    def calc_obstacle_cost(self, trajectory, ob):
        """
        calc obstacle cost inf: collision
        """
        # 장애물 좌표 추출
        ox = ob[:, 0]
        oy = ob[:, 1]

        # 장애물까지의 거리 계산
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
            if (np.logical_and(np.logical_and(upper_check, right_check),
                            np.logical_and(bottom_check, left_check))).any():
                return float("Inf")
        elif self.robot_type == RobotType.circle:
            if np.array(r <= self.robot_radius*1.5).any():
                return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK

    def calc_terrain_cost(self, trajectory, terrain_polygons):
        """
        Calculate the cost associated with terrain polygons.
        """
        cost = 0.0
        for polygon_points in terrain_polygons:
            polygon = Polygon(polygon_points)
            for point in trajectory[:, :2]:
                p = Point(point[0], point[1])
                if polygon.contains(p):
                    cost += float("Inf")  # Collision cost
                else:
                    cost += polygon.exterior.distance(p)
        return cost/1000

    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]

        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def euclidean_distance(self, x, y):
        return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def cal_to_goal_dist(self,trajectory, goal):
        h = self.euclidean_distance([trajectory[-1,0], trajectory[-1,1]], [goal[0],goal[1]])
        return h

    def calc_control_and_trajectory(self, x, dw, goal, ob, terrain_polygons):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)
                # calc cost
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)
                terrain_cost = self.calc_terrain_cost(trajectory, terrain_polygons)
                heuristic = self.heuristic_cost_gain * self.cal_to_goal_dist(trajectory, goal)

                final_cost = to_goal_cost + speed_cost + ob_cost + heuristic + terrain_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(x[3]) < self.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory

    def funcInternalTransition(self):

        self.dist_to_goal = math.hypot(self.curPose[0] - self.goal[0], self.curPose[1] - self.goal[1])
        if self.dist_to_goal <= self.target_tolerance:
            print("Goal reached.")
            self.setStateValue("mode", 'ARRIVE')
        else:
            dw = self.calc_dynamic_window(self.curPose, self.goal)
            u, trajectory = self.calc_control_and_trajectory(self.curPose, dw, self.goal, self.ob, self.terrain_polygons)
            self.setStateValue("target_linearVelocity", u[0])
            self.setStateValue("target_angularVelocity", u[1])

        # dw = self.calc_dynamic_window(self.curPose, self.goal)
        # u, trajectory = self.calc_control_and_trajectory(self.curPose, dw, self.goal, self.ob, self.terrain_polygons)
        # self.setStateValue("target_linearVelocity", u[0])
        # self.setStateValue("target_angularVelocity", u[1])
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "PLANNING" or self.getStateValue('mode') ==('ARRIVE'):
            return 0.1
        else:
            return 999999999
    
    def funcSelect(self):
        pass
