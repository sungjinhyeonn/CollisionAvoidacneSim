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
import pandas as pd
from collections import defaultdict
from stable_baselines3 import SAC

class STGPModel(nn.Module):
    def __init__(self, input_dim, num_steps=4):
        super(STGPModel, self).__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=4, padding=4)
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2 * num_steps)  # 각 스텝별 (x, y) 예측

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.fc(x[:, -1, :])  # [batch_size, num_steps * 2]
        return x.reshape(-1, self.num_steps, 2)  # [batch_size, num_steps, 2]
    
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = None
        print(f"Initialized SACAgent with state_dim={state_dim}, action_dim={action_dim}")

    def load_model(self, model_path):
        """GYM에서 학습한 SAC 모델 로드"""
        try:
            # 모델 경로 확인
            print(f"Attempting to load model from: {model_path}")
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                return False

            # 모델 로드 시도
            print("Loading SAC model...")
            self.model = SAC.load(model_path)
            print("Model loaded successfully")
            
            # 모델 속성 확인
            print(f"Model type: {type(self.model)}")
            print(f"Model observation space: {self.model.observation_space}")
            print(f"Model action space: {self.model.action_space}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            return False

    def predict(self, state, deterministic=True):
        """상태를 입력받아 행동 예측"""
        try:
            if self.model is None:
                print("Error: Model is not loaded")
                return None, None
            
            # 상태 전처리
            if isinstance(state, list):
                state = np.array(state)
            elif isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            
            # 상태 형태 확인 및 출력
            print(f"Input state shape: {state.shape}")
            print(f"Input state: {state}")
            
            # 상태가 1차원 배열이면 2차원으로 변환
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            
            # 상태 차원 확인
            if state.shape[1] != self.state_dim:
                print(f"State dimension mismatch. Expected {self.state_dim}, got {state.shape[1]}")
                return None, None
            
            # 모델 예측 수행
            try:
                action, _states = self.model.predict(state, deterministic=deterministic)
                print(f"Predicted action: {action}")
                return action, _states
            except Exception as e:
                print(f"Model prediction error: {e}")
                import traceback
                traceback.print_exc()
                return None, None
                
        except Exception as e:
            print(f"Error in predict method: {e}")
            import traceback
            traceback.print_exc()
            return None, None

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
        self.ray_length = 10.0  # max_ray_length와 동일
        state_dim = 4 + self.num_rays  # 상태 차원도 그에 맞게 수정 (로봇 위치(2) + 목표 정보(2) + 레이(16))
        action_dim = 2     # 선속도, 각속도
        
        self.drl_gym_agent = SACAgent(state_dim, action_dim)
        self.drl_agent = DDPGAgent(state_dim, action_dim)
        # 알고리즘 설정 가져오기
        self.use_drl = objConfiguration.getConfiguration('use_drl') or False
        self.use_dwa = objConfiguration.getConfiguration('use_dwa') or False
        self.use_apf = objConfiguration.getConfiguration('use_apf') or False
        self.use_sac = objConfiguration.getConfiguration('use_sac') or False
        self.sac_model_path = objConfiguration.getConfiguration('sac_model_path')
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
        elif self.use_sac:
            try:
                # Configuration에서 모델 경로 가져오기
                model_path = self.objConfiguration.getConfiguration('sac_model_path')
                
                if not os.path.exists(model_path):
                    print(f"Model not found at specified path: {model_path}")
                    return
                    
                print(f"Loading SAC model from: {model_path}")
                
                # SACAgent 초기화 및 모델 로드
                self.drl_gym_agent = SACAgent(state_dim=4+self.num_rays, action_dim=2)
                self.drl_gym_agent.load_model(model_path)
                
            except Exception as e:
                print(f"Error loading SAC model: {e}")
                import traceback
                traceback.print_exc()
        # elif self.use_sac:
        #     try:
        #         # 모델 디렉토리 설정
                
        #         models_dir = r"C:\Users\User\Desktop\CDE\(241219)Pathplanning framework\A_SAC\models"
        #         model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
                
        #         if not model_files:
        #             print("No trained models found!")
        #             return
                    
        #         # 가장 최근 모델 선택
        #         latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        #         model_path = os.path.join(models_dir, latest_model)
        #         print(f"Loading latest model: {model_path}")
                
        #         # SACAgent 초기화 및 모델 로드
        #         self.drl_gym_agent = SACAgent(state_dim=4+self.num_rays, action_dim=2)
        #         self.drl_gym_agent.load_model(model_path)
                
        #     except Exception as e:
        #         print(f"Error loading GYM model: {e}")
        #         import traceback
        #         traceback.print_exc()
        # DRL 모드 추가
        self.ray_length = 5.0  # 레이캐스트 최대 거리

        # DRL 모드 관련 변수 추가
        self.drl_mode_start_time = 0
        self.drl_mode_timeout = 15.0  # 5초에서 15초로 증가
        self.current_time = 0
        self.min_obstacle_distance = 1.0  # 장애물이 이 거리보다 멀어지면 REPLAN으로 전환

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

        # 장애물 예측 관련 변수 추가
        self.obstacle_history = defaultdict(list)  # 각 장애물의 이동 기록
        self.sequence_length = 10  # 예측에 사용할 시퀀스 길이
        self.prediction_model = None
        
        # DTP 관련 설정
        self.use_dtp = objConfiguration.getConfiguration('use_dtp') or False
        self.dtp_model_path = objConfiguration.getConfiguration('dtp_model_path')
        self.visualize_dtp = objConfiguration.getConfiguration('visualize_dtp') or False
        
        # 시각화를 위한 플롯 설정
        if self.visualize_dtp:
            plt.ion()  # 대화형 모드 활성화
            self.fig_dtp, self.ax_dtp = plt.subplots()
            self.ax_dtp.set_title('DTP Prediction Visualization')
            self.prediction_plots = {}  # 각 장애물별 예측 플롯 저장

        # DTP가 활성화된 경우에만 모델 로드
        if self.use_dtp:
            self.load_prediction_model()

    def load_prediction_model(self):
        """예측 모델 로드"""
        try:
            model_path = 'DTP/trained_stgp_model.pth'  # 모델 경로
            self.prediction_model = STGPModel(input_dim=5, num_steps=2)  # 2스텝으로 변경
            self.prediction_model.load_state_dict(torch.load(model_path))
            self.prediction_model.eval()
            print("Obstacle prediction model loaded successfully")
        except Exception as e:
            print(f"Failed to load prediction model: {e}")
            self.prediction_model = None

    def update_obstacle_history(self, obstacle_id, position, dt):
        """장애물의 이동 기록 업데이트"""
        history = self.obstacle_history[obstacle_id]
        if not history:
            # 첫 데이터인 경우
            history.append({
                'x': position[0],
                'y': position[1],
                'yaw': 0.0,  # 초기값
                'linear_velocity': 0.0,
                'angular_velocity': 0.0,
                'timestamp': dt
            })
        else:
            # 속도 계산
            prev = history[-1]
            dx = position[0] - prev['x']
            dy = position[1] - prev['y']
            dt = dt - prev['timestamp']
            if dt > 0:
                linear_velocity = np.sqrt(dx*dx + dy*dy) / dt
                yaw = np.arctan2(dy, dx)
                angular_velocity = (yaw - prev['yaw']) / dt if len(history) > 1 else 0.0
            else:
                linear_velocity = prev['linear_velocity']
                yaw = prev['yaw']
                angular_velocity = prev['angular_velocity']

            history.append({
                'x': position[0],
                'y': position[1],
                'yaw': yaw,
                'linear_velocity': linear_velocity,
                'angular_velocity': angular_velocity,
                'timestamp': dt
            })

        # 히스토리 크기 제한
        if len(history) > self.sequence_length:
            history.pop(0)
        self.obstacle_history[obstacle_id] = history

    def predict_obstacle_position(self, obstacle_id):
        """장애물의 다음 4스텝 위치 예측"""
        if self.prediction_model is None or len(self.obstacle_history[obstacle_id]) < self.sequence_length:
            return None

        try:
            # 시퀀스 데이터 준비
            history = self.obstacle_history[obstacle_id]
            sequence = np.array([[d['x'], d['y'], d['yaw'], d['linear_velocity'], d['angular_velocity']] 
                               for d in history[-self.sequence_length:]])
            
            # 모델 입력 형태로 변환
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0)

            # 예측 수행
            with torch.no_grad():
                predicted_positions = self.prediction_model(input_tensor)
                predicted_positions = predicted_positions.squeeze().numpy()  # [num_steps, 2]

            return predicted_positions

        except Exception as e:
            print(f"Prediction error for obstacle {obstacle_id}: {e}")
            return None

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
                obstacle_pos = (objEvent.dblPositionE, objEvent.dblPositionN)
                
                if self.use_dtp:
                    # DTP가 활성화된 경우 예측 수행
                    self.update_obstacle_history(objEvent.strID, obstacle_pos, self.simulation_time)
                    predicted_pos = self.predict_obstacle_position(objEvent.strID)
                    if predicted_pos is not None:
                        self.obstacles[objEvent.strID] = (predicted_pos[0], predicted_pos[1])
                    else:
                        self.obstacles[objEvent.strID] = obstacle_pos
                    
                    # 모든 장애물의 예측 결과를 한번에 시각화
                    if self.visualize_dtp:
                        self.visualize_all_predictions()
                else:
                    self.obstacles[objEvent.strID] = obstacle_pos

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
            elif self.use_sac:
                target_state = self.calc_sac_action()
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
                # 장애물이 충분히 멀어졌으면 리랜
                print("Obstacle is far enough, switching to REPLAN")
                self.setStateValue("mode", "REPLAN")
                done_message = MsgManeuverState(self.ID, self.curPose[1], self.curPose[0])
                self.addOutputEvent("Replan", done_message)
            else:
                # 아직 장애물이 가까이 있으면 DRL 계속 사용
                self.setStateValue("mode", "DRL")

        elif mode == "REPLAN":
            # 리플랜 요청 후 커로 돌아감
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

    def calc_sac_action(self):

            # SAC용 상태 정보 가져오기
            state = self.get_sac_state()
            if state is None:
                print("Failed to get SAC state")
                return None
            
            print(f"Current position: x={self.curPose[0]:.2f}, y={self.curPose[1]:.2f}")
            
            # SAC 모델로부터 행동 예측
            action_result = self.drl_gym_agent.predict(state)
            print(f"Raw action result: {action_result}")
            
            if action_result is None or action_result[0] is None:
                print("Invalid action result")
                return None
            
            # 2차원 배열에서 1차원 배열로 변환
            action = action_result[0].squeeze()  # 차원 축소
            print(f"Processed action: {action}")
            
            if not isinstance(action, (list, np.ndarray)) or len(action) != 2:
                print(f"Unexpected action format: {action}")
                return None
            
            # 행동(선속도, 각속도)을 다음 상태로 변환
            linear_vel = float(action[0])
            angular_vel = float(action[1])
            
            print(f"Parsed velocities: linear={linear_vel}, angular={angular_vel}")
            
            dt = self.objConfiguration.getConfiguration("dt") or 0.1
            
            # 현재 상태에서 다음 상태 계산
            current_x, current_y, current_yaw = self.curPose[:3]
            next_yaw = current_yaw + angular_vel * dt
            next_x = current_x + linear_vel * np.cos(next_yaw) * dt
            next_y = current_y + linear_vel * np.sin(next_yaw) * dt
            
            target_state = [next_x, next_y, next_yaw]
            
            # 디버깅을 위한 출력
            print(f"GYM DRL action: linear_vel={linear_vel:.2f}, angular_vel={angular_vel:.2f}")
            print(f"GYM DRL target: x={target_state[0]:.2f}, y={target_state[1]:.2f}, yaw={target_state[2]:.2f}")
            
            # 상태 업데이트
            self.setStateValue("target_x", target_state[0])
            self.setStateValue("target_y", target_state[1])
            self.setStateValue("target_yaw", target_state[2])
            
            return target_state


    def calc_drl_action(self):
        """DRL 기반 충돌 회피 행동 계산"""
        try:
            # DRL 상태 벡터 획득
            state = self.get_drl_state()
            if state is None:
                return None
            
            # 상태 정규화 (실제 환경 → 학습 환경 스케일)
            normalized_state = self.normalize_state(state)
            
            # DRL 행동 결정
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
                
                # 장애물과의 거리에 따른 가중치 계산
                obstacle_distance = self.check_obstacle_distance()
                distance_to_goal = self.check_distance()
                
                # 장애물 거리에 따른 속도 가중치 (0.3 ~ 1.2)
                obstacle_weight = np.clip(obstacle_distance / 10.0, 0.3, 1.2)
                
                # 목표 거리에 따른 속도 가중치 (0.5 ~ 1.5)
                goal_weight = np.clip(1.5 - (distance_to_goal / 20.0), 0.5, 1.5)
                
                # 최종 가중치 계산
                speed_weight = min(obstacle_weight, goal_weight)
                turn_weight = 1.0  # 기본 회전 가중치
                
                # 장애물이 매우 가까울 경우 회전 가중치 증가
                if obstacle_distance < 5.0:
                    turn_weight = 0.7
                
                # 목표가 가까울 경우 속도 감소
                if distance_to_goal < 3.0:
                    speed_weight *= 0.7
                
                # 가중치 적용된 액션
                weighted_action = np.array([
                    action[0] * speed_weight,  # 선속도에 가중치 적용
                    action[1] * speed_weight    # 각속도에 가중치 적용
                ])
                
                # 액션 클리핑 (-1 ~ 1 범위 유지)
                weighted_action = np.clip(weighted_action, -1, 1)
                
                # 실제 환경 스케일로 변환
                real_action = self.denormalize_action(weighted_action, state)
                
                # 기본값 설정
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
                action = self.drl_agent.act(state_tensor)  # 상대적 이동량
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                if len(action.shape) > 1:
                    action = action.squeeze()
                
                # DRL 출력값 확인
                print(f"Raw DRL action: {action}")
                
                if action is None or len(action) != 2:
                    print("Invalid DRL output")
                    return None
                
                # 현재 위치에 상대적 이동량 더하기
                current_pos = np.array([self.curPose[0], self.curPose[1]])
                next_pos = current_pos + action  # 상대적 이동량을 현재 위치에 더함
                
                # 이동 벡터 확인
                movement_vector = action  # 이동 벡터는 DRL 출력 그대로
                movement_distance = np.linalg.norm(movement_vector)
                print(f"Movement vector: {movement_vector}, Distance: {movement_distance:.2f}")
                
                # 다음 방향 계산
                next_yaw = np.arctan2(movement_vector[1], movement_vector[0])
                target_state = [next_pos[0], next_pos[1], next_yaw]
                
                # 목표 위치 출력
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
        """레이캐스트 수행 함수 - DTP 사용 여부에 따라 다르게 동작"""
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
        
        # 현재 장애물 위치와 예측된 위치를 모두 고려
        all_obstacle_positions = []
        for obs_id, obs_pos in self.obstacles.items():
            # 현재 위치 추가
            all_obstacle_positions.append(np.array(obs_pos))
            
            # DTP가 활성화된 경우에만 예측 위치 추가
            if self.use_dtp:
                predicted_pos = self.predict_obstacle_position(obs_id)
                if predicted_pos is not None:
                    all_obstacle_positions.append(predicted_pos)
        
        if not all_obstacle_positions:
            self.cached_rays = ray_distances
            self.last_pose = self.curPose.copy()
            self.last_obstacles = obstacles_tuple
            return ray_distances
        
        obstacle_positions = np.array(all_obstacle_positions)
        
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
                collision_threshold = robot_radius + 1.0  # 1.0은 장애물의 기본 반경
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
    
    def get_sac_state(self):
        """SAC 에이전트를 위한 상태 벡터 생성 (8개 레이)"""
        if self.goal is None:
            return None
        
        try:
            # 현재 위치를 원점으로 하는 상대 좌표로 변환
            current_pos = np.array(self.curPose[:2])
            goal_pos = np.array(self.goal)
            relative_goal = goal_pos - current_pos
            
            # 목표까지의 거리와 각도 계산
            distance_to_goal = np.linalg.norm(relative_goal)
            angle_to_goal = np.arctan2(relative_goal[1], relative_goal[0]) - self.curPose[2]
            angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
            
            # 8개의 레이캐스트 수행
            rays = self.cast_rays()
            ray_distances = []
            
            for i in range(self.num_rays):  # self.num_rays = 8
                min_distance = self.ray_length
                ray_angle = self.curPose[2] + i * (2 * np.pi / self.num_rays)
                ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
                
                for obstacle_pos in self.obstacles.values():
                    obstacle_pos = np.array(obstacle_pos)
                    to_obstacle = obstacle_pos - current_pos
                    distance_along_ray = np.dot(to_obstacle, ray_direction)
                    
                    if 0 < distance_along_ray < self.ray_length:
                        lateral_distance = np.abs(np.cross(ray_direction, to_obstacle))
                        if lateral_distance < 1.0:
                            min_distance = min(min_distance, distance_along_ray)
            
                ray_distances.append(min_distance)
            
            # 상태 벡터 구성: [로봇 위치(2), 목표 정보(2), 레이(8)]
            state = np.concatenate([
                [0.0, 0.0],              # 로봇 중심 기준
                [distance_to_goal],       # 목표까지의 거리
                [angle_to_goal],          # 목표까지의 각도
                ray_distances             # 8개의 레이캐스트 거리
            ])
            
            print(f"SAC state vector shape: {state.shape}")  # 디버깅용
            return state
            
        except Exception as e:
            print(f"Error in get_sac_state: {e}")
            return None

    def get_drl_state(self):
        """DRL 상태 벡터 획득 (로컬 좌표계 기준)"""
        try:
            if not self.curPose or not self.goal:
                return None

            # 현재 로봇의 위치와 방향
            robot_x = self.curPose[0]
            robot_y = self.curPose[1]
            robot_yaw = self.curPose[2]

            # 글로벌 좌표를 로컬 좌표로 변환하는 함수
            def to_local_coord(global_x, global_y):
                # 평행이동
                dx = global_x - robot_x
                dy = global_y - robot_y
                
                # 회전 변환
                local_x = dx * np.cos(-robot_yaw) - dy * np.sin(-robot_yaw)
                local_y = dx * np.sin(-robot_yaw) + dy * np.cos(-robot_yaw)
                
                return local_x, local_y

            # 목표점을 로컬 좌표로 변환
            goal_local_x, goal_local_y = to_local_coord(self.goal[0], self.goal[1])
            
            # 목표까지의 거리와 각도 계산 (로컬 좌표 기준)
            distance_to_goal = np.sqrt(goal_local_x**2 + goal_local_y**2)
            angle_to_goal = np.arctan2(goal_local_y, goal_local_x)

            # 장애물 정보를 로컬 좌표로 변환
            local_obstacles = []
            for obs_pos in self.obstacles.values():
                local_x, local_y = to_local_coord(obs_pos[0], obs_pos[1])
                local_obstacles.append([local_x, local_y])

            # 레이캐스트 거리 계산 (로컬 좌표 기준)
            ray_distances = self.cast_rays()  # 이미 로봇 중심 기준으로 계산됨

            # 상태 벡터 구성
            state = np.zeros(4 + len(ray_distances))  # 기본 상태 + 레이캐스트
            state[0] = 0  # 로컬 좌표계에서 로봇은 원점
            state[1] = 0
            state[2] = distance_to_goal
            state[3] = angle_to_goal
            state[4:] = ray_distances

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

    def visualize_all_predictions(self):
        """모든 장애물의 예측 결과를 한번에 시각화"""
        if not self.visualize_dtp:
            return
        
        try:
            # 현재 시간 체크하여 업데이트 간격 조절
            current_time = time.time()
            if hasattr(self, 'last_plot_time') and current_time - self.last_plot_time < 0.5:
                return
            self.last_plot_time = current_time

            # 플롯 초기화
            self.ax_dtp.clear()
            
            # 현재 로봇 위치 표시
            self.ax_dtp.plot(self.curPose[0], self.curPose[1], 'bo', markersize=10, label='Robot')
            
            # 목표 위치 표시
            if self.goal:
                self.ax_dtp.plot(self.goal[0], self.goal[1], 'g*', markersize=15, label='Goal')
            
            # 모든 장애물의 현재 위치와 예측 위치 표시
            for obs_id, obs_pos in self.obstacles.items():
                current_pos = np.array(obs_pos)
                predicted_positions = self.predict_obstacle_position(obs_id)
                
                # 현재 위치 표시 (빨간색 원)
                self.ax_dtp.plot(current_pos[0], current_pos[1], 'ro', markersize=10, 
                               label=f'Current Position' if obs_id == list(self.obstacles.keys())[0] else "")
                
                # 예측 경로 표시
                if predicted_positions is not None:
                    # 예측 경로를 점선으로 표시
                    pred_x = [current_pos[0]] + list(predicted_positions[:, 0])
                    pred_y = [current_pos[1]] + list(predicted_positions[:, 1])
                    self.ax_dtp.plot(pred_x, pred_y, '--b', alpha=0.5,
                                   label=f'Predicted Path' if obs_id == list(self.obstacles.keys())[0] else "")
                    
                    # 각 예측 위치를 점으로 표시
                    for step, pred_pos in enumerate(predicted_positions, 1):
                        self.ax_dtp.plot(pred_pos[0], pred_pos[1], 'bo', markersize=8, alpha=0.5)
                        self.ax_dtp.text(pred_pos[0], pred_pos[1]-0.5, f't+{step}',
                                       bbox=dict(facecolor='white', alpha=0.7))
                    
                    # 마지막 예측 위치까지의 거리 표시
                    total_dist = np.linalg.norm(predicted_positions[-1] - current_pos)
                    self.ax_dtp.text(predicted_positions[-1][0], predicted_positions[-1][1]+0.5, 
                                   f'Total: {total_dist:.2f}m',
                                   bbox=dict(facecolor='white', alpha=0.7))
                
                # 장애물 ID 표시
                self.ax_dtp.text(current_pos[0], current_pos[1]-1, f'{obs_id}',
                               bbox=dict(facecolor='white', alpha=0.7))
            
            # 그래프 설정
            self.ax_dtp.set_xlim([0, 30])
            self.ax_dtp.set_ylim([0, 30])
            self.ax_dtp.grid(True)
            self.ax_dtp.legend(loc='upper left', bbox_to_anchor=(1, 1))
            self.ax_dtp.set_title(f'DTP Prediction (Time: {self.simulation_time:.1f}s)')
            
            # 축 레이블 추가
            self.ax_dtp.set_xlabel('X Position (m)')
            self.ax_dtp.set_ylabel('Y Position (m)')
            
            # 그래프 여백 조정 (범례가 잘리지 않도록)
            plt.tight_layout()
            
            # 그프 업데이트
            self.fig_dtp.canvas.draw()
            self.fig_dtp.canvas.flush_events()
            
        except Exception as e:
            print(f"Visualization error: {e}")

    def __del__(self):
        """소멸자에서 플롯 정리"""
        try:
            if self.visualize_dtp:
                plt.close(self.fig_dtp)
        except:
            pass