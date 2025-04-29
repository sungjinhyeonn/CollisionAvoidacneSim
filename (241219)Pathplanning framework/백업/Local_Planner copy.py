from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgManeuverState import MsgManeuverState
from shapely.geometry import Polygon, Point
import numpy as np
import math
from enum import Enum

class RobotType(Enum):
    circle = 0
    rectangle = 1

class LPP(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objConfiguration = objConfiguration
        self.terrain_polygons = objConfiguration.getConfiguration('terrain_polygons')
        self.safety_tolerance = objConfiguration.getConfiguration('safety_tolerance')

        self.goal = None
        self.path = []
        self.curPose = [0, 0, 0, 0, 0]  # 현재 위치: x, y, yaw, lin_vel, ang_vel
        self.obstacles = {}  # 장애물 정보를 저장할 딕셔너리
        self.other_agents = {}  # 다른 에이전트의 위치 정보를 저장할 딕셔너리
        self.ob = np.array([])  # 장애물 위치 정보 배열로 저장

        # 상태 변수
        self.addStateVariable("mode", "CHECKER")
        self.addStateVariable("ta", float('inf'))  # 기본 상태는 무한 대기 상태

        # 입력 포트
        self.addInputPort("GlobalWaypoint")  # GPP로부터 받은 전역 경유지
        self.addInputPort("OtherManeuverState")  # 다른 이동체의 상태
        self.addInputPort("MyManeuverState")  # 본인의 상태
        self.addInputPort("stopSim")  # 시뮬레이션 종료 신호

        # 출력 포트
        self.addOutputPort("RequestManeuver")  # 이동 요청
        
        self.addOutputPort("Replan")  # 재계획 요청
        self.robot_type = RobotType.rectangle
        self.robot_length = 1
        self.robot_width = 1
        self.setStateValue("target_x", None)
        self.setStateValue("target_y", None)


    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "MyManeuverState":
            # 현재 위치 업데이트
            self.curPose[0] = objEvent.x
            self.curPose[1] = objEvent.y
            self.curPose[2] = objEvent.yaw
            self.curPose[3] = objEvent.lin_vel
            self.curPose[4] = objEvent.ang_vel
            print("Updated current pose:", self.curPose)
            self.continueTimeAdvance()

        elif strPort == "OtherManeuverState" and self.getStateValue("mode") != 'ARRIVE':
            # 다른 이동체 및 장애물 정보 업데이트
            if "Obstacle" in objEvent.strID:
                self.obstacles[objEvent.strID] = (objEvent.dblPositionN, objEvent.dblPositionE)
            else:
                if objEvent.strID[:8] != self.ID[:8]:  # 본인의 ID를 제외
                    self.other_agents[objEvent.strID] = (objEvent.dblPositionN, objEvent.dblPositionE)
                    # 다른 에이전트의 위치를 장애물로 간주하여 추가
                    self.obstacles[objEvent.strID] = (objEvent.dblPositionN, objEvent.dblPositionE)

            # 장애물 위치 정보를 배열로 업데이트
            self.ob = np.array(list(self.obstacles.values()))
            self.continueTimeAdvance()

        elif strPort == "GlobalWaypoint":
            # 목적지 설정 이벤트
            self.path = objEvent
            self.goal = self.path  # 마지막 점을 목표 지점으로 설정
            self.setStateValue("mode", "CHECKER")

            self.continueTimeAdvance()

        elif strPort == "stopSim":
            # 시뮬레이션 종료 시 초기화
            self.path = []
            self.goal = None
            self.obstacles.clear()
            self.other_agents.clear()
            self.setStateValue("mode", "WAIT")


        return True
    
    def funcInternalTransition(self):
        mode = self.getStateValue("mode")

        if mode == "CHECKER":
            if self.goal:
                # 목표 위치와 현재 위치 간의 거리 계산
                distance_to_goal = self.check_distance()
                print(f"Distance to goal: {distance_to_goal}")

                if distance_to_goal < self.objConfiguration.getConfiguration("target_tolerance"):
                    # 목표에 도달한 경우
                    print("Arrived at goal")
                    self.setStateValue("mode", "REPLAN")
    
                else:
                    # 아직 목표에 도달하지 않은 경우 DWA 상태로 전환
                    print("Switching to DWA mode")
                    self.setStateValue("mode", "DWA")

            else:
                # 목표가 없는 경우 REPLAN 상태로 전환
                print("No goal available. Switching to REPLAN mode")
                # self.setStateValue("mode", "REPLAN")


        elif mode == "DWA":
            # DWA 기반으로 다음 위치 계산
            self.calc_dwa()
            self.setStateValue("mode", "CHECKER")
 

        elif mode == "ARRIVE":
            # 목표 도착 후 Done 메시지 전송
           pass


        elif mode == "REPLAN":
            # Replan 메시지 전송
            done_message = MsgManeuverState(self.ID, self.curPose[1], self.curPose[0])
            print("Replanning")
            self.addOutputEvent("Replan", done_message)
            self.setStateValue("mode", "CHECKER")
      

        return True


    def funcOutput(self):
        if self.getStateValue("mode") == "DWA" and self.getStateValue("target_x") != None:
            objRequestMessage = MsgManeuverState(self.ID, self.getStateValue("target_x"), self.getStateValue("target_y"))
            self.addOutputEvent("RequestManeuver", objRequestMessage)

        return True

    def funcTimeAdvance(self):
        mode = self.getStateValue("mode")
        
        if mode == "CHECKER":
            # CHECKER 상태에서는 1초 대기 후 전이
            return 1.0
        elif mode == "DWA":
            # DWA 상태에서는 빠르게 전이 (0.1초)
            return 1
        elif mode == "ARRIVE":
            # ARRIVE 상태에서는 즉시 전이
            return 0.0
        elif mode == "REPLAN":
            # REPLAN 상태에서는 즉시 전이
            return 0.0
        elif mode == "WAIT":
            # WAIT 상태에서는 무한 대기
            return float('inf')
        else:
            # 기본 대기 시간 (예: 5초)
            return 1


    def calc_dwa(self):
        """DWA 기반 지역 경로 계획 수행"""
        # Dynamic Window 생성
        dw = self.calc_dynamic_window(self.curPose)
        
        # 최적의 제어 입력과 경로를 선택
        best_u, best_trajectory = self.calc_control_and_trajectory(self.curPose, dw, self.goal, self.obstacles, self.terrain_polygons)

        # 경로에서 가장 마지막 위치를 다음 목표 위치로 설정
        self.curPose = best_trajectory[-1].tolist()
        self.setStateValue("target_x", self.curPose[0])
        self.setStateValue("target_y", self.curPose[1])
        self.setStateValue("target_yaw", self.curPose[2])

    def calc_dynamic_window(self, x):
        Vs = [self.objConfiguration.getConfiguration('min_speed'),
            self.objConfiguration.getConfiguration('max_speed'),
            -self.objConfiguration.getConfiguration('max_yaw_rate'),
            self.objConfiguration.getConfiguration('max_yaw_rate')]

        Vd = [x[3] - self.objConfiguration.getConfiguration('max_accel') * self.objConfiguration.getConfiguration('dt'),
            x[3] + self.objConfiguration.getConfiguration('max_accel') * self.objConfiguration.getConfiguration('dt'),
            x[4] - self.objConfiguration.getConfiguration('max_delta_yaw_rate') * self.objConfiguration.getConfiguration('dt'),
            x[4] + self.objConfiguration.getConfiguration('max_delta_yaw_rate') * self.objConfiguration.getConfiguration('dt')]

        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        # 디버깅 출력
        # print(f"Dynamic Window: {dw}")
        return dw


    def calc_control_and_trajectory(self, x, dw, goal, ob, terrain_polygons):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        for v in np.arange(dw[0], dw[1], self.objConfiguration.getConfiguration('v_resolution')):
            for y in np.arange(dw[2], dw[3], self.objConfiguration.getConfiguration('yaw_rate_resolution')):

                trajectory = self.predict_trajectory(x_init, v, y)
                to_goal_cost = self.objConfiguration.getConfiguration('to_goal_cost_gain') * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.objConfiguration.getConfiguration('speed_cost_gain') * (self.objConfiguration.getConfiguration('max_speed') - trajectory[-1, 3])
                ob_cost = self.objConfiguration.getConfiguration('obstacle_cost_gain') * self.calc_obstacle_cost(trajectory, ob)
                heuristic = self.objConfiguration.getConfiguration('heuristic_cost_gain') * self.calc_to_goal_dist(trajectory, goal)
                # terrian_cost = self.calc_terrain_cost(trajectory, terrain_polygons)

                final_cost = to_goal_cost + speed_cost + ob_cost + heuristic 

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory

        return best_u, best_trajectory

    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.objConfiguration.getConfiguration('predict_time'):
            x = self.motion(x, [v, y], self.objConfiguration.getConfiguration('dt'))
            trajectory = np.vstack((trajectory, x))
            time += self.objConfiguration.getConfiguration('dt')
        return trajectory

    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x
    
    def check_distance(self):
        """
        현재 위치와 목표 위치 간의 유클리드 거리 계산.
        """
        if self.curPose and self.goal:
            current_position = np.array(self.curPose[:2])
            goal_position = np.array([self.goal.dblPositionE, self.goal.dblPositionN])
            distance = np.linalg.norm(current_position - goal_position)
            return distance
        return float("inf")

    def check_obstacle_distance(self):
        # 장애물과의 거리 계산
        if not self.curPose:
            return float('inf')
        # 장애물 계산 로직 구현
        return float('inf')

    def update_data(self, objEvent):
        # 센서 또는 외부로부터 받은 상태 정보 업데이트
        pass
    def calc_to_goal_cost(self, trajectory, goal):
        # if not trajectory.any() or len(trajectory) == 0:
        #     raise ValueError("Trajectory is empty or undefined.")
        # if not goal or len(goal) != 2:
        #     raise ValueError("Goal position must be a tuple or list of two elements (x, y).")
        
        # Calculate the direction to the goal
        dx = goal.dblPositionE - trajectory[-1, 0]
        dy = goal.dblPositionN - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)  # Direction from current position to goal

        # Calculate the difference angle between the agent's current heading and the goal direction
        current_heading = trajectory[-1, 2]
        cost_angle = error_angle - current_heading
        
        # Normalize the cost angle to be within the range of -pi to pi
        cost_angle = (cost_angle + math.pi) % (2 * math.pi) - math.pi

        # Calculate the absolute minimal angle difference
        cost = min(cost_angle, 2 * math.pi - abs(cost_angle))

        return abs(cost)
        
    def calc_obstacle_cost(self, trajectory, ob):
        # Ensure `ob` is a NumPy array
        if not isinstance(ob, np.ndarray):
            ob = np.array(ob) if isinstance(ob, list) else np.array(list(ob.values()))

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


    def calc_to_goal_dist(self, trajectory, goal):
        h = self.euclidean_distance([trajectory[-1, 0], trajectory[-1, 1]], [goal.dblPositionE, goal.dblPositionN])
        return h
    
    
    def euclidean_distance(self, x, y):
        return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))