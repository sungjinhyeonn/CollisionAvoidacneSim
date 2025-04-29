from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgManeuverState import MsgManeuverState
from Models.Atomic.DWA import DWAPlanner
import numpy as np
from matplotlib.patches import Polygon as pltPolygon
from shapely.geometry import Point, LineString, Polygon
import torch
import torch.nn as nn
from DRL.DRL import Actor, Critic, DDPGAgent
import matplotlib.pyplot as plt
import time
import os
import math

class LPP(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        # Configuration
        self.objConfiguration = objConfiguration
        self.terrain_polygons = objConfiguration.getConfiguration('terrain_polygons')
        self.safety_tolerance = objConfiguration.getConfiguration('safety_tolerance')
        
        # State variables
        self.goal = None
        self.path = []
        self.curPose = [0, 0, 0, 0, 0]  # x, y, yaw, lin_vel, ang_vel
        self.obstacles = {}
        self.other_agents = {}
        self.ob = np.array([])
        self.predict_ta = objConfiguration.getConfiguration('predict_time')

        # Initialize DWA Planner
        self.dwa = DWAPlanner(objConfiguration)

        # DEVS State Variables
        self.addStateVariable("mode", "CHECKER")
        self.addStateVariable("ta", float('inf'))
        self.setStateValue("target_x", None)
        self.setStateValue("target_y", None)

        # Input Ports
        self.addInputPort("GlobalWaypoint")
        self.addInputPort("OtherManeuverState")
        self.addInputPort("MyManeuverState")
        self.addInputPort("StopSimulation_IN")

        # Output Ports
        self.addOutputPort("RequestManeuver")
        self.addOutputPort("Replan")

        self.dwa_stuck_counter = 0
        self.last_positions = []
        self.max_stuck_count = 15
        self.position_history_size = 8
        self.escape_mode = False
        self.escape_direction = None
        self.escape_attempts = 0
        self.max_escape_attempts = 5

        # DRL 관련 변수 수정
        self.num_rays = 8     # env.py와 동일하게 16개로 변경
        self.ray_length = 25.0  # max_ray_length와 동일
        state_dim = 4 + self.num_rays  # 상태 차원도 그에 맞게 수정 (로봇 위치(2) + 목표 정보(2) + 레이(16))
        action_dim = 2     # 선속도, 각속도
        self.drl_agent = DDPGAgent(state_dim, action_dim)
        
        # 알고리즘 설정 가져오기
        self.use_drl = objConfiguration.getConfiguration('use_drl') or False
        self.use_dwa = objConfiguration.getConfiguration('use_dwa') or False
        self.use_apf = objConfiguration.getConfiguration('use_apf') or False
        
        # DRL 모델 로드
        if self.use_drl:
            checkpoint_path = objConfiguration.getConfiguration('drl_checkpoint')
            try:
                checkpoint = torch.load(checkpoint_path)
                self.drl_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.drl_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                print(f"Loaded DRL model from {checkpoint_path}")
            except:
                print("No pre-trained model found, using new model")

        # DRL 모드 추가
        self.ray_length = 5.0  # 레이캐스트 최대 거리

        # DRL 모드 관련 변수 추가
        self.drl_mode_start_time = 0
        self.drl_mode_timeout = 15.0  # 5초에서 15초로 증가
        self.current_time = 0
        self.min_obstacle_distance = 3.0  # 장애물이 이 거리보다 멀어지면 REPLAN으로 전환

        # 시각화 설정

        self.visualization_enabled = True  # 기본값 True
        
        # 시각화 업데이트 간격 설정
        self.last_viz_update = 0
        self.viz_update_interval = 0.1
        self.simulation_time = 0.0  # simulation_time 변수 추가

        # 캐시 추가
        self.cached_rays = None
        self.last_pose = None
        self.last_obstacles = None

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "MyManeuverState":
            self.curPose[0] = objEvent.x
            self.curPose[1] = objEvent.y
            self.curPose[2] = objEvent.yaw
            self.curPose[3] = objEvent.lin_vel
            self.curPose[4] = objEvent.ang_vel
            print("Updated current pose:", self.curPose)
            self.continueTimeAdvance()

        elif strPort == "OtherManeuverState" and self.getStateValue("mode") != 'ARRIVE':
            if "Obstacle" in objEvent.strID:
                # 장애물 위치 업데이
                self.obstacles[objEvent.strID] = (objEvent.dblPositionE, objEvent.dblPositionN)
                
            else:
                if objEvent.strID[:8] != self.ID[:8]:
                    self.other_agents[objEvent.strID] = (objEvent.dblPositionE, objEvent.dblPositionN)
                    if self.check_agent_distance(objEvent.strID) < self.safety_tolerance:
                        self.obstacles[objEvent.strID] = (objEvent.dblPositionE, objEvent.dblPositionN)
                        if self.getStateValue("mode") != "DWA":
                            print(f"Agent {objEvent.strID} detected nearby! Switching to DWA mode")
                            self.setStateValue("mode", "DWA")
                    elif objEvent.strID in self.obstacles:
                        del self.obstacles[objEvent.strID]
            self.ob = np.array(list(self.obstacles.values()))
            self.continueTimeAdvance()

        elif strPort == "GlobalWaypoint":
            if self.path != [objEvent.dblPositionE, objEvent.dblPositionN]:
                print("First waypoint received")
                self.path = objEvent
                self.goal = (objEvent.dblPositionE, objEvent.dblPositionN)
                self.setStateValue("mode", "SEND")
            self.continueTimeAdvance()

        elif strPort == "StopSimulation_IN":
            print(f"{self.ID}: Received StopSimulation message")
            self.setStateValue("mode", "WAIT")  # DEAD -> WAIT
            self.continueTimeAdvance()
            return True

        # 상태 업데이트마다 로깅 제거 (funcTimeAdvance에서 수행)
        # self.log_state()

        return True

    def funcInternalTransition(self):
        mode = self.getStateValue("mode")

        if mode == "SEND":
            # 첫 웨이포인트를 보낸 후 커 상태로
            self.setStateValue("mode", "CHECKER")

        elif mode == "CHECKER":
            if not self.goal:
                print("No goal available. Switching to REPLAN mode")
                self.setStateValue("mode", "REPLAN")
                return True

            distance_to_goal = self.check_distance()
            obstacle_distance = self.check_obstacle_distance()
            
            print(f"Distance to goal: {distance_to_goal}")
            print(f"Obstacle distance: {obstacle_distance}")
            print(f"Safety tolerance: {self.safety_tolerance}")

            if distance_to_goal < self.objConfiguration.getConfiguration("target_tolerance"):
                print("Arrived at goal")
                self.setStateValue("mode", "REPLAN")
                done_message = MsgManeuverState(self.ID, self.curPose[1], self.curPose[0])
                self.addOutputEvent("Replan", done_message)
            else:
                if obstacle_distance < self.safety_tolerance:
                    print(f"Obstacle detected at distance {obstacle_distance}")
                    self.setStateValue("mode", "DRL")
                else:
                    print(f"No nearby obstacles (distance: {obstacle_distance})")

        elif mode == "DRL":
            # 위치 히스토리 업데이트
            current_pos = np.array(self.curPose[:2])
            self.last_positions.append(current_pos)
            if len(self.last_positions) > self.position_history_size:
                self.last_positions.pop(0)

            # DWA 알고리즘 사용
            if self.use_dwa:
                target_state = self.calc_dwa()
            elif self.use_drl:
                target_state = self.calc_drl_action_enu()
            elif self.use_apf:   
                target_state = self.calc_apf_action()  # APF로 전환하려면 이 줄의 주석을 해제
            
            if target_state is None:
                print("Action calculation failed, switching to REPLAN")
                self.setStateValue("mode", "REPLAN")
                return True
                
            # 액션 실행을 위해 SEND_LOCAL로 전환
            self.setStateValue("mode", "SEND_LOCAL")

        elif mode == "SEND_LOCAL":
            # 애물이 충분히 멀어졌는지 체크
            if self.check_obstacle_distance() > self.min_obstacle_distance:
                # 장애물이 충분히 멀어졌으면 리플랜
                print("Obstacle is far enough, switching to REPLAN")
                self.setStateValue("mode", "REPLAN")
                done_message = MsgManeuverState(self.ID, self.curPose[1], self.curPose[0])
                self.addOutputEvent("Replan", done_message)
            else:
                # 아직 장애물이 가까이 있으면 DRL 계속 사용
                self.setStateValue("mode", "DRL")

        elif mode == "REPLAN":
            # 리플랜 요청 후 체커로 돌아감
            print("Replanning")
            done_message = MsgManeuverState(self.ID, self.curPose[1], self.curPose[0])
            self.addOutputEvent("Replan", done_message)
            self.setStateValue("mode", "CHECKER")

        # 모든 상태 변경 후 로깅 제거 (funcTimeAdvance에서 수행)
        # self.log_state()

        return True

    def funcOutput(self):
        mode = self.getStateValue("mode")

        if mode == "SEND" and self.path:
            x, y = self.path.dblPositionE, self.path.dblPositionN
            print(f"Sending first waypoint: ({x}, {y})")
            objRequestMessage = MsgManeuverState(self.ID, y, x)
            self.addOutputEvent("RequestManeuver", objRequestMessage)

        elif mode == "SEND_LOCAL":
            objRequestMessage = MsgManeuverState(
                self.ID, 
                self.getStateValue("target_y"), 
                self.getStateValue("target_x")
            )
            print(f"Sending DWA waypoint: ({self.getStateValue('target_x')}, {self.getStateValue('target_y')})")
            self.addOutputEvent("RequestManeuver", objRequestMessage)

        return True

    def funcTimeAdvance(self):
        mode = self.getStateValue("mode")
        dt = self.objConfiguration.getConfiguration("dt") or 0.1
        self.simulation_time += dt
        
        # 상태별 시간 반환
        if mode == "WAIT":
            return 9999999999999
        elif mode in ["SEND", "SEND_LOCAL", "REPLAN"]:
            return 0.0
        elif mode == "CHECKER":
            return 0.1
        elif mode == "DRL":
            return dt
        else:
            return 1.0

    def calc_dwa(self):
        """DWA 기반 지역 경로 계획 수행"""
        target_state = self.dwa.calc_dwa(
            self.curPose,
            self.goal,
            self.obstacles,
            self.terrain_polygons
        )
                        # 상태 업데이트
        self.setStateValue("target_x", target_state[0])
        self.setStateValue("target_y", target_state[1])
        self.setStateValue("target_yaw", target_state[2])
        
        return target_state
    
    def normalize_state(self, state):
        """실제 환경 좌표를 학습 환경 좌표로 정규화 (0~100 → 0~20)"""
        normalized_state = state.copy()
        
        # 현재 위치 정규화 (x, y)
        normalized_state[0] = state[0] / 5 # 100 → 20 스케일로 환
        normalized_state[1] = state[1] / 5
        
        # 목표까지의 거리 정규화
        normalized_state[2] = state[2] / 5 # distance_to_goal
        
        # 각도는 미 -pi~pi로 정규화되 있으므로 그대로 사용
        # normalized_state[3] = state[3]  # angle_to_goal
        
        # 레이캐스트 거리 정규화
        for i in range(4, len(state)):  # ray distances
            normalized_state[i] = state[i] / 5
        
        return normalized_state

    def denormalize_action(self, action, current_state):
        """학습 환경의 션을 실제 환경의 케일로 환"""
        real_action = action.copy()
        # 속도 이미 -1~1로 정규화되어 있으므로 그대로 사용
        return real_action

    def calc_drl_action(self):
        """DRL 기반 충돌 회피 행동 계산"""
        try:
            # DRL 상태 벡터 기
            state = self.get_drl_state()
            if state is None:
                return None
            
            # 상 정규화 (실제 환경 → 학습 환경 스케일)
            normalized_state = self.normalize_state(state)
            
            # DRL 동 결정
            with torch.no_grad():
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                action = self.drl_agent.act(state_tensor)
                
                # numpy 배열이 아닌 경우 변환
                if not isinstance(action, np.ndarray):
                    action = action.cpu().numpy()
                
                # 원 맞지 않는 우 처리
                if len(action.shape) > 1:
                    action = action.squeeze()
                
                # 액션 유효성 검사
                if action is None or len(action) != 2:
                    return None
                
                # 액션 실제 환경 스케로 변환
                real_action = self.denormalize_action(action, state)
                
                # 기본값 정
                max_speed = self.objConfiguration.getConfiguration("max_speed") or 1.0
                max_yawrate = self.objConfiguration.getConfiguration("max_yawrate") or np.pi/4
                dt = self.objConfiguration.getConfiguration("dt") or 0.1
                
                # 행동을 로봇 제어 명령으 변환
                linear_vel = np.clip(real_action[0], -1, 1) * max_speed
                angular_vel = np.clip(real_action[1], -1, 1) * max_yawrate
                
                # 다음 상태 예측
                next_x = self.curPose[0] + linear_vel * np.cos(self.curPose[2]) * dt
                next_y = self.curPose[1] + linear_vel * np.sin(self.curPose[2]) * dt
                next_yaw = self.curPose[2] + angular_vel * dt
                
                target_state = [next_x, next_y, next_yaw]
                
                # 상태 업데이트
                self.setStateValue("target_x", target_state[0])
                self.setStateValue("target_y", target_state[1])
                self.setStateValue("target_yaw", target_state[2])
                
                return target_state
                
        except Exception as e:
            return None
        
    def calc_drl_action_enu(self):
        try:
            state = self.get_drl_state()
            if state is None:
                print("Failed to get DRL state")
                return None
            
            # 현재 위치 출력
            print(f"Current position: x={self.curPose[0]:.2f}, y={self.curPose[1]:.2f}")
            
            normalized_state = self.normalize_state(state)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                next_pos = self.drl_agent.act(state_tensor)
                
                if isinstance(next_pos, torch.Tensor):
                    next_pos = next_pos.cpu().numpy()
                
                if len(next_pos.shape) > 1:
                    next_pos = next_pos.squeeze()
                
                # DRL 출력값 확인
                print(f"Raw DRL output: {next_pos}")
                
                if next_pos is None or len(next_pos) != 2:
                    print("Invalid DRL output")
                    return None
                
                # 이동 벡터 확인
                current_pos = np.array([self.curPose[0], self.curPose[1]])
                to_next = next_pos - current_pos
                movement_distance = np.linalg.norm(to_next)
                print(f"Movement vector: {to_next}, Distance: {movement_distance:.2f}")
                
                next_yaw = np.arctan2(to_next[1], to_next[0])
                target_state = [next_pos[0], next_pos[1], next_yaw]
                
                # 종 목표 위치 출력
                print(f"DRL target: x={target_state[0]:.2f}, y={target_state[1]:.2f}, yaw={target_state[2]:.2f}")
                            # 상태 업데이트
                self.setStateValue("target_x", target_state[0])
                self.setStateValue("target_y", target_state[1])
                self.setStateValue("target_yaw", target_state[2])
                
                return target_state
                
        except Exception as e:
            print(f"DRL action calculation error: {e}")
            return None

    def check_distance(self):
        """현재 위치와 목표 위치 사이의 거리를 계산"""
        if not self.curPose or not self.goal:
            return float('inf')
        
        current_position = np.array(self.curPose[:2])
        goal_position = np.array(self.goal)
        distance = np.linalg.norm(goal_position - current_position)
        
        return distance

    def check_obstacle_distance(self):
        """현재 위치와 가장 가까운 장애물 사이의 거리를 계산"""
        if not self.curPose or not self.obstacles:
            return float('inf')
        
        current_position = np.array(self.curPose[:2])
        min_distance = float('inf')
        
        for obstacle_pos in self.obstacles.values():
            obstacle_position = np.array(obstacle_pos)
            distance = np.linalg.norm(obstacle_position - current_position)
            if distance < min_distance:
                min_distance = distance
                
        print(f"Closest obstacle distance: {min_distance}")  # 디버깅을 위한 출력
        return min_distance

    def check_terrain_distance(self):
        if not self.curPose or not self.terrain_polygons:
            return float('inf')

        current_position = Point(self.curPose[0], self.curPose[1])
        min_distance = float('inf')

        for polygon_points in self.terrain_polygons:
            polygon = Polygon(polygon_points)
            distance = polygon.exterior.distance(current_position)
            min_distance = min(min_distance, distance)

        return min_distance

    def is_stuck_in_dwa(self):
        """DWA 상태에서 stuck 여부를 확인"""
        if len(self.last_positions) < self.position_history_size:
            return False
        
        total_step_distance = 0
        for i in range(1, len(self.last_positions)):
            pos1 = np.array(self.last_positions[i-1])
            pos2 = np.array(self.last_positions[i])
            total_step_distance += np.linalg.norm(pos2 - pos1)
        
        start_pos = np.array(self.last_positions[0])
        current_pos = np.array(self.last_positions[-1])
        direct_distance = np.linalg.norm(current_pos - start_pos)
        
        is_oscillating = total_step_distance > direct_distance * 1.3
        is_not_moving = total_step_distance < 0.5
        
        return is_oscillating or is_not_moving

    def generate_escape_point(self):
        """Stuck 상태에서 벗어나기 위한 임시 목점 생성"""
        current_pos = np.array(self.curPose[:2])
        goal_pos = np.array(self.goal)
        
        to_goal = goal_pos - current_pos
        to_goal_norm = np.linalg.norm(to_goal)
        
        if to_goal_norm < 0.1:
            return None
        
        base_escape_distance = 8.0
        escape_distance = base_escape_distance * (1 + self.escape_attempts * 0.8)
        
        if self.obstacles:
            obstacle_positions = np.array(list(self.obstacles.values()))
            
            angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            candidates = []
            for angle in angles:
                rad = np.radians(angle)
                direction = np.array([np.cos(rad), np.sin(rad)])
                point = current_pos + direction * escape_distance
                candidates.append(point)
            
            min_distances = []
            for point in candidates:
                distances = np.linalg.norm(obstacle_positions - point, axis=1)
                min_distances.append(np.min(distances))
            
            sorted_indices = np.argsort(min_distances)[::-1]
            best_candidates = sorted_indices[:3]
            
            min_goal_distance = float('inf')
            best_point = None
            for idx in best_candidates:
                point = candidates[idx]
                goal_distance = np.linalg.norm(point - goal_pos)
                if goal_distance < min_goal_distance:
                    min_goal_distance = goal_distance
                    best_point = point
            
            escape_point = best_point
        else:
            escape_point = current_pos + to_goal / to_goal_norm * escape_distance
        
        self.escape_attempts += 1
        return escape_point

    def check_agent_distance(self, agent_id):
        """다른 에이전트와의 거리 """
        if not self.curPose or agent_id not in self.other_agents:
            return float('inf')
        
        current_position = np.array(self.curPose[:2])
        agent_position = np.array(self.other_agents[agent_id])
        return np.linalg.norm(agent_position - current_position)

    def cast_rays(self):
        """레이캐스트 수행 함수 - 로봇 크기 반영"""
        # 캐시 체크
        current_pos = np.array(self.curPose[:2])
        current_yaw = self.curPose[2]
        obstacles_tuple = tuple(sorted(self.obstacles.items()))
        
        if (self.last_pose is not None and 
            np.allclose(current_pos, self.last_pose[:2]) and 
            np.isclose(current_yaw, self.last_pose[2]) and 
            obstacles_tuple == self.last_obstacles):
            return self.cached_rays
        
        ray_distances = np.full(self.num_rays, self.ray_length)
        
        # 로봇 반경 가져오기
        robot_radius = self.objConfiguration.getConfiguration("robot_radius") or 3.0
        
        # 장애물 위치를 numpy 배열 변환
        obstacle_positions = np.array(list(self.obstacles.values()))
        
        if len(obstacle_positions) == 0:
            self.cached_rays = ray_distances
            self.last_pose = self.curPose.copy()
            self.last_obstacles = obstacles_tuple
            return ray_distances
        
        # 각도 계산을 벡터화
        angles = current_yaw + np.arange(self.num_rays) * (2 * np.pi / self.num_rays)
        ray_directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        # 각 레이에 대해
        for i, ray_dir in enumerate(ray_directions):
            # 모든 장애물에 대한 계산을 벡터화
            to_obstacles = obstacle_positions - current_pos
            distances_along_ray = np.dot(to_obstacles, ray_dir)
            mask = (0 < distances_along_ray) & (distances_along_ray < self.ray_length)
            
            if np.any(mask):
                # 유효한 장애물에 대해서만 계산
                valid_obstacles = to_obstacles[mask]
                valid_distances = distances_along_ray[mask]
                lateral_distances = np.abs(np.cross(ray_dir, valid_obstacles))
                
                # 로봇 크기와 장애물 크기를 고려한 충돌 검사
                collision_threshold = robot_radius + 1.0  # 1.0은 장애물의 본 반경
                collision_mask = lateral_distances < collision_threshold
                
                if np.any(collision_mask):
                    # 충돌 지점까지의 거리에서 로봇 반경을 뺌
                    collision_distances = valid_distances[collision_mask]
                    min_distance = np.min(collision_distances)
                    ray_distances[i] = max(0.0, min_distance - robot_radius)
        
        # 캐시 업데이트
        self.cached_rays = ray_distances
        self.last_pose = self.curPose.copy()
        self.last_obstacles = obstacles_tuple
        
        return ray_distances

    def get_drl_state(self):
        """DRL 에전트를 위한 상태 벡터 생성"""
        if self.goal is None:
            return None
        
        try:
            # 현재 위치를 원점으 하는 상대 좌표로 변환
            current_pos = np.array(self.curPose[:2])
            goal_pos = np.array(self.goal)
            relative_goal = goal_pos - current_pos
            
            # 목표까지의 거리와 각 계산
            distance_to_goal = np.linalg.norm(relative_goal)
            angle_to_goal = np.arctan2(relative_goal[1], relative_goal[0]) - self.curPose[2]
            angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi  # -pi에서 pi 범위로 정규화
            
            # 레이캐스 수행
            rays = self.cast_rays()
            ray_distances = []
            
            for i in range(self.num_rays):
                min_distance = self.ray_length
                ray_angle = self.curPose[2] + i * (2 * np.pi / self.num_rays)
                ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
                
                # 장애물과의 거리 계산
                for obstacle_pos in self.obstacles.values():
                    obstacle_pos = np.array(obstacle_pos)
                    to_obstacle = obstacle_pos - current_pos
                    
                    # 레 방향으로의 투영 거리 계산
                    distance_along_ray = np.dot(to_obstacle, ray_direction)
                    
                    if 0 < distance_along_ray < self.ray_length:
                        # 레이 수직인 거리 계산
                        lateral_distance = np.abs(np.cross(ray_direction, to_obstacle))
                        
                        if lateral_distance < 1.0:  # 장애물이 레이와 충분 가까운 경우
                            min_distance = min(min_distance, distance_along_ray)
                
                ray_distances.append(min_distance)
            
            # 태 벡터 구: [local_x, local_y, distance_to_goal, angle_to_goal, ray1, ray2, ..., ray8]
            state = np.concatenate([
                [0.0, 0.0],              # 로봇 중심 기준 (항상 0,0)
                [distance_to_goal],       # 목표지의 거리
                [angle_to_goal],          # 목표까지의 도
                ray_distances             # 레이캐스트 거리
            ])
            
            return state
            
        except Exception as e:
            print(f"Error in get_drl_state: {e}")
            return None

    def calc_apf_action(self):
        """APF 기반 충돌 회피 행동 계산"""
        try:
            # APF 파라미터 조정
            k_att = 1.0      
            k_rep = 50.0    # 척력 계수 감소
            rho_0 = 8.0      
            
            # 현재 위치
            curr_pos = np.array([self.curPose[0], self.curPose[1]])
            curr_yaw = self.curPose[2]
            
            # 인 계산 (단위 벡터로 정규화)
            to_goal = self.goal - curr_pos
            dist_to_goal = np.linalg.norm(to_goal)
            att_force = k_att * to_goal / dist_to_goal
            
            # 척력 계산 (지수적으로 증가)
            rep_force = np.zeros(2)
            min_obstacle_dist = float('inf')
            
            for obs_pos in self.obstacles.values():
                obs = np.array(obs_pos)
                to_robot = curr_pos - obs
                dist = np.linalg.norm(to_robot)
                min_obstacle_dist = min(min_obstacle_dist, dist)
                
                if dist <= rho_0:
                    n = to_robot / dist
                    # 더 부드러운 척력 함수 사용
                    rep_magnitude = k_rep * (1.0/dist - 1.0/rho_0)  # 지수 함수 제거
                    rep_force += rep_magnitude * n
            
            # 가까울 때의 증폭도 더 완만게
            if min_obstacle_dist < rho_0/2:
                rep_force *= (rho_0/min_obstacle_dist)  # 제곱 대신 선형 증가
            
            # 전체 힘의 가중치도 조정
            total_force = att_force + 2.0 * rep_force  # 5.0에서 2.0으로 감소
            
            # 방향 계산
            desired_yaw = math.atan2(total_force[1], total_force[0])
            yaw_diff = math.atan2(math.sin(desired_yaw - curr_yaw),
                                 math.cos(desired_yaw - curr_yaw))
            
            # 각속도 계산
            max_yaw_rate = self.objConfiguration.getConfiguration("max_yaw_rate") or (10 * math.pi / 180.0)
            angular_vel = np.clip(5.0 * yaw_diff, -max_yaw_rate, max_yaw_rate)
            
            # 선속도 계산
            max_speed = self.objConfiguration.getConfiguration("max_speed") or 1.0
            min_speed = 0.2
            
            if min_obstacle_dist < rho_0:
                speed_factor = (min_obstacle_dist / rho_0) ** 2
                linear_vel = max(min_speed, max_speed * speed_factor)
            else:
                linear_vel = max_speed
            
            # 다음 상태 계산
            dt = self.objConfiguration.getConfiguration("dt") or 0.2
            next_yaw = curr_yaw + angular_vel * dt
            next_x = curr_pos[0] + linear_vel * math.cos(next_yaw) * dt
            next_y = curr_pos[1] + linear_vel * math.sin(next_yaw) * dt
            
            target_state = [next_x, next_y, next_yaw]
            
            # 상태 업데이트
            self.setStateValue("target_x", target_state[0])
            self.setStateValue("target_y", target_state[1])
            self.setStateValue("target_yaw", target_state[2])
            
            return target_state
            
        except Exception as e:
            print(f"APF calculation error: {e}")
            return None

    def __del__(self):
        """소멸자에서 파 리"""
        try:
            if self.fig:
                plt.close(self.fig)
        except:
            pass