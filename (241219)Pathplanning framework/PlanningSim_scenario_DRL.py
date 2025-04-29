import os
import math
from enum import Enum
from SimulationEngine.SimulationEngine import SimulationEngine
from SimulationEngine.Utility.Configurator import Configurator
from Models.Outmost import Outmost
import numpy as np
from warehouse_layout import LAYOUT
import random
from obstacle_scenarios import SCENARIOS
import matplotlib.pyplot as plt
import pandas as pd
# 기존의 terrain_polygons, picking_zones 등의 정의를 LAYOUT에서 가져오기
terrain_polygons = LAYOUT['storage_racks']
picking_zones = LAYOUT['picking_zones']

# 시나리오 선택 (1~5)
scenario_num = 2  # 원하는 시나리오 번호 선택
selected_scenario = SCENARIOS[f'scenario_{scenario_num}']

# 장애물 설정 - 시나리오에서 가져오기
obstacle_positions = selected_scenario['positions']
obstacle_targets = selected_scenario['targets']

objConfiguration = Configurator()

class RobotType(Enum):
    circle = 0
    rectangle = 1

# 파라미터 설정
objConfiguration.addConfiguration('scenario', scenario_num)
objConfiguration.addConfiguration('robot_radius', 2)
objConfiguration.addConfiguration('max_speed', 1.0)
objConfiguration.addConfiguration('min_speed', 0.0)
objConfiguration.addConfiguration('max_yaw_rate', 20 * math.pi / 180.0)
objConfiguration.addConfiguration('max_accel', 0.2)
objConfiguration.addConfiguration('max_delta_yaw_rate', 360.0 * math.pi / 180.0)
objConfiguration.addConfiguration('v_resolution', 0.1)
objConfiguration.addConfiguration('yaw_rate_resolution', 1 * math.pi / 180.0)
objConfiguration.addConfiguration('dt', 0.5)
objConfiguration.addConfiguration('predict_time', 3)
objConfiguration.addConfiguration('to_goal_cost_gain', 1.0)
objConfiguration.addConfiguration('speed_cost_gain', 1.0)
objConfiguration.addConfiguration('obstacle_cost_gain', 50.0)
objConfiguration.addConfiguration('robot_type', RobotType.circle)
objConfiguration.addConfiguration('robot_stuck_flag_cons', 2)
objConfiguration.addConfiguration('target_tolerance', 1.0)
objConfiguration.addConfiguration('safety_tolerance', 5)
objConfiguration.addConfiguration('terrain_tolerance', 2)
objConfiguration.addConfiguration('slowdown_distance', 2)
objConfiguration.addConfiguration('obstacle_radius', 0.7)
objConfiguration.addConfiguration('use_drl', False)  # DDPG 사용
objConfiguration.addConfiguration('use_dwa', False)  # DWA 미사용
objConfiguration.addConfiguration('use_apf', False)  # APF 미사용
objConfiguration.addConfiguration('use_gym_drl', True)  # gym_drl 사용
objConfiguration.addConfiguration('drl_checkpoint', 'DRL/best_model_ENU3.pth')
objConfiguration.addConfiguration('gym_model_path', 'GYM/models/sac_robot_interrupted_20241224_174637/sac_robot_interrupted_20241224_174637.zip')
# DTP(Dynamic Trajectory Prediction) 관련 설정
objConfiguration.addConfiguration('use_dtp', False)  # DTP 사용
objConfiguration.addConfiguration('dtp_model_path', 'DTP/trained_stgp_model.pth')
objConfiguration.addConfiguration('visualize_dtp', False)

dt = objConfiguration.getConfiguration('dt')

# AMR 시작 위치와 목적지 설정 (20x20 맵)
agent_starts = [
    [0.0, 0.0, math.pi/4, 0.0, 0.0]
]

agent_goals = [
    [
        [17.0, 17.0],  # 맵 크기에 맞게 목적지 수정
        [18.0, 18.0]
    ]
]

# 설정 업데이트
objConfiguration.addConfiguration('numAgent', len(agent_starts))
objConfiguration.addConfiguration('agent_starts', agent_starts)
objConfiguration.addConfiguration('agent_goals', agent_goals)
objConfiguration.addConfiguration('obstacle_targets', obstacle_targets)
objConfiguration.addConfiguration('numObstacles', len(obstacle_positions))
objConfiguration.addConfiguration('obstacle_positions', obstacle_positions)

# 장애물 이동 속도 설정 - 좀 더 자연스러운 움직임을 위해 조정
objConfiguration.addConfiguration('obstacle_speed', 0.1)  # [m/s]

# 설정 정보 업데이트
objConfiguration.addConfiguration('obstacle_yaws', [0.0] * len(obstacle_positions))

# 지형 데이터 설정 (선 구조물)
objConfiguration.addConfiguration('terrain_polygons', terrain_polygons)
objConfiguration.addConfiguration('picking_zones', picking_zones)
def analyze_results(log_dir):
    """결과 분석 함수"""
    # Agent로 시작하는 모든 CSV 파일 찾기
    agent_files = [f for f in os.listdir(log_dir) if f.startswith('Agent_') and f.endswith('.csv')]
    if not agent_files:
        print(f"No agent data files found in {log_dir}")
        return None
        
    # 가장 최근 파일 사용
    latest_file = sorted(agent_files)[-1]
    agent_data = pd.read_csv(f'{log_dir}/{latest_file}')
    
    # 분석 메트릭 계산
    total_distance = 0
    collision_count = 0
    min_obstacle_distance = float('inf')
    
    for i in range(1, len(agent_data)):
        # 이동 거리 계산
        dx = agent_data['x'].iloc[i] - agent_data['x'].iloc[i-1]
        dy = agent_data['y'].iloc[i] - agent_data['y'].iloc[i-1]
        total_distance += np.sqrt(dx*dx + dy*dy)
        
        # 장애물과의 최소 거리 업데이트
        if 'min_obstacle_distance' in agent_data.columns:
            min_obstacle_distance = min(min_obstacle_distance, 
                                     agent_data['min_obstacle_distance'].iloc[i])
    
    # 목표 도달 여부 확인
    final_pos = np.array([agent_data['x'].iloc[-1], agent_data['y'].iloc[-1]])
    goal_pos = np.array([17.0, 17.0])  # 목표 위치
    reached_goal = np.linalg.norm(final_pos - goal_pos) < 1.0
    
    # 운행 시간 계산 (마지막 타임스탬프)
    operation_time = agent_data['timestamp'].iloc[-1]
    
    return {
        'total_distance': total_distance,
        'collision_count': collision_count,
        'min_obstacle_distance': min_obstacle_distance,
        'reached_goal': reached_goal,
        'execution_time': len(agent_data) * 0.1,  # dt = 0.1
        'operation_time': operation_time  # 실제 운행 시간 추가
    }

def visualize_trajectory(log_dir):
    """경로 시각화 함수"""
    # Agent로 시작하는 모든 CSV 파일 찾기
    agent_files = [f for f in os.listdir(log_dir) if f.startswith('Agent_') and f.endswith('.csv')]
    if not agent_files:
        print(f"No agent data files found in {log_dir}")
        return
        
    # 가장 최근 파일 사용
    latest_file = sorted(agent_files)[-1]
    agent_data = pd.read_csv(f'{log_dir}/{latest_file}')
    
    plt.figure(figsize=(10, 10))
    plt.plot(agent_data['x'], agent_data['y'], 'b-', label='Robot Path')
    plt.plot(0, 0, 'go', label='Start')
    plt.plot(17, 17, 'r*', label='Goal')
    
    # 장애물 위치 표시
    for pos in SCENARIOS['scenario_2']['positions']:  # scenario_num에 맞게 수정
        plt.plot(pos[0], pos[1], 'k.', markersize=10)
    
    plt.grid(True)
    plt.legend()
    plt.title('Robot Trajectory')
    plt.axis('equal')
    plt.savefig(f'{log_dir}/trajectory.png')
    plt.close()

def visualize_trajectory_with_obstacles(log_dir):
    """시간에 따른 에이전트와 장애물의 위치를 시각화"""
    # 에이전트 데이터 읽기
    agent_files = [f for f in os.listdir(log_dir) if f.startswith('Agent_') and f.endswith('.csv')]
    if not agent_files:
        print(f"No agent data files found in {log_dir}")
        return
    latest_agent_file = sorted(agent_files)[-1]
    agent_data = pd.read_csv(f'{log_dir}/{latest_agent_file}')
    
    # 장애물 데이터 읽기
    obstacle_files = [f for f in os.listdir(log_dir) if f.startswith('Obstacle_') and f.endswith('.csv')]
    obstacle_data = {}
    for obs_file in obstacle_files:
        obs_id = obs_file.split('_')[1]  # Obstacle_0 -> 0
        obstacle_data[obs_id] = pd.read_csv(f'{log_dir}/{obs_file}')
    
    # 시각화
    plt.figure(figsize=(15, 15))
    
    # 에이전트 경로 그리기
    plt.plot(agent_data['x'], agent_data['y'], 'b-', label='Robot Path', alpha=0.5)
    
    # 시작점과 목표점
    plt.plot(0, 0, 'go', label='Start', markersize=15)
    plt.plot(17, 17, 'r*', label='Goal', markersize=15)
    
    # 시간 간격으로 위치 표시
    time_interval = len(agent_data) // 10  # 10개의 포인트만 표시
    for i in range(0, len(agent_data), time_interval):
        # 에이전트 위치
        plt.plot(agent_data['x'].iloc[i], agent_data['y'].iloc[i], 'bo', alpha=0.5)
        plt.text(agent_data['x'].iloc[i], agent_data['y'].iloc[i], 
                f't={agent_data["timestamp"].iloc[i]:.1f}s', 
                fontsize=8)
        
        # 각 장애물의 해당 시점 위치
        for obs_id, obs_data in obstacle_data.items():
            if i < len(obs_data):
                plt.plot(obs_data['x'].iloc[i], obs_data['y'].iloc[i], 
                        'ro', alpha=0.5, markersize=8)
                plt.text(obs_data['x'].iloc[i], obs_data['y'].iloc[i],
                        f'Obs{obs_id}', fontsize=8)
    
    # 장애물의 경로도 표시
    for obs_id, obs_data in obstacle_data.items():
        plt.plot(obs_data['x'], obs_data['y'], 'r--', 
                alpha=0.3, label=f'Obstacle {obs_id} Path')
    
    plt.grid(True)
    plt.legend()
    plt.title('Robot and Obstacles Trajectories Over Time')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.savefig(f'{log_dir}/trajectory_with_obstacles.png', dpi=300, bbox_inches='tight')
    plt.close()
# 시뮬레이션 모델 설정
objModels = Outmost(objConfiguration)
engine = SimulationEngine()
engine.setOutmostModel(objModels)

if __name__ == "__main__":
    try:
        print("Starting simulation with DDPG + DTP...")
        algorithm = "APF"
        log_dir = f'log/scenario_{scenario_num}/{algorithm}'
        engine.run(maxTime=9999999, \
                logFileName='log.txt', \
                visualizer=False, \
                logGeneral=False, \
                logActivateState=False, \
                logActivateMessage=False, \
                logActivateTA=False, \
                logStructure=False \
                )
        visualize_trajectory(log_dir)
        visualize_trajectory_with_obstacles(log_dir)
    except Exception as e:
        print(f"Error during simulation: {e}")
