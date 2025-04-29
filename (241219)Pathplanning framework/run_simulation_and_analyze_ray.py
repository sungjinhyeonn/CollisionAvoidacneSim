import os
import math
import numpy as np
from enum import Enum
from SimulationEngine.SimulationEngine import SimulationEngine
from SimulationEngine.Utility.Configurator import Configurator
from Models.Outmost import Outmost
from warehouse_layout import LAYOUT
from obstacle_scenarios import SCENARIOS
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle
import time

class RobotType(Enum):
    circle = 0
    rectangle = 1

def run_simulation(algorithm_type, scenario_num=1):
    """시뮬레이션 실행 함수"""
    # 기본 설정
    terrain_polygons = LAYOUT['storage_racks']
    picking_zones = LAYOUT['picking_zones']
    
    # 시나리오 선택
    selected_scenario = SCENARIOS[f'scenario_{scenario_num}']
    obstacle_positions = selected_scenario['positions']
    obstacle_targets = selected_scenario['targets']
    
    # Configuration 설정
    objConfiguration = Configurator()
    
    # 기본 파라미터 설정
    objConfiguration.addConfiguration('robot_radius', 1)
    objConfiguration.addConfiguration('max_speed', 1.5)
    objConfiguration.addConfiguration('min_speed', 0.0)
    objConfiguration.addConfiguration('max_yaw_rate', 40 * math.pi / 180.0)
    objConfiguration.addConfiguration('max_accel', 0.2)
    objConfiguration.addConfiguration('max_delta_yaw_rate', 360.0 * math.pi / 180.0)
    objConfiguration.addConfiguration('v_resolution', 0.1)
    objConfiguration.addConfiguration('yaw_rate_resolution', 1 * math.pi / 180.0)
    objConfiguration.addConfiguration('dt', 1)
    objConfiguration.addConfiguration('predict_time', 3)
    objConfiguration.addConfiguration('to_goal_cost_gain', 1.0)
    objConfiguration.addConfiguration('speed_cost_gain', 1.0)
    objConfiguration.addConfiguration('obstacle_cost_gain', 5.0)
    objConfiguration.addConfiguration('robot_type', RobotType.circle)
    objConfiguration.addConfiguration('robot_stuck_flag_cons', 2)
    objConfiguration.addConfiguration('target_tolerance', 1.0)
    objConfiguration.addConfiguration('safety_tolerance',8)
    objConfiguration.addConfiguration('terrain_tolerance', 2)
    objConfiguration.addConfiguration('slowdown_distance', 2)
    objConfiguration.addConfiguration('obstacle_radius', 0.7)
    
    # 알고리즘별 설정
    if algorithm_type == "DDPG_DTP":

        objConfiguration.addConfiguration('use_drl', True)
        objConfiguration.addConfiguration('use_dwa', False)
        objConfiguration.addConfiguration('use_dtp', True)
        objConfiguration.addConfiguration('drl_checkpoint', 'DRL/best_model_7.pth')
        objConfiguration.addConfiguration('drl_type', 'DDPG')
    elif algorithm_type == "DDPG":

        objConfiguration.addConfiguration('use_drl', True)
        objConfiguration.addConfiguration('use_dwa', False)
        objConfiguration.addConfiguration('use_dtp', False)
        objConfiguration.addConfiguration('drl_checkpoint', 'DRL/best_model_7.pth')
        objConfiguration.addConfiguration('drl_type', 'DDPG')
    elif algorithm_type == "DWA":
 
        objConfiguration.addConfiguration('use_drl', False)
        objConfiguration.addConfiguration('use_dwa', True)
        objConfiguration.addConfiguration('use_dtp', False)
    
    # 시작점과 목표점 설정
    agent_starts = [[0.0, 0.0, math.pi/4, 0.0, 0.0]]
    agent_goals = [[[17.0, 17.0], [18.0, 18.0]]]
    
    # Configuration 업데이트
    objConfiguration.addConfiguration('numAgent', len(agent_starts))
    objConfiguration.addConfiguration('agent_starts', agent_starts)
    objConfiguration.addConfiguration('agent_goals', agent_goals)
    objConfiguration.addConfiguration('obstacle_targets', obstacle_targets)
    objConfiguration.addConfiguration('numObstacles', len(obstacle_positions))
    objConfiguration.addConfiguration('obstacle_positions', obstacle_positions)
    
    # 장애물 관련 설정 추가
    objConfiguration.addConfiguration('obstacle_speed', 0.1)
    objConfiguration.addConfiguration('obstacle_yaws', [0.0] * len(obstacle_positions))
    
    # 지형 데이터 설정
    objConfiguration.addConfiguration('terrain_polygons', terrain_polygons)
    objConfiguration.addConfiguration('picking_zones', picking_zones)
    
    # 시뮬레이션 실행
    print(f"\nStarting simulation with {algorithm_type}...")
    start_time = time.time()
    
    objModels = Outmost(objConfiguration)
    engine = SimulationEngine()
    engine.setOutmostModel(objModels)
    
    log_dir = f'log/{algorithm_type}'
    os.makedirs(log_dir, exist_ok=True)
    
    engine.run(maxTime=9999999,
              logFileName='log.txt',
              visualizer=False,
              logGeneral=False,
              logActivateState=False,
              logActivateMessage=False,
              logActivateTA=False,
              logStructure=False)
    
    execution_time = time.time() - start_time
    return log_dir, execution_time

def analyze_results(log_dir):
    """결과 분석 함수"""
    try:
        # Agent로 시작하는 모든 CSV 파일 찾기
        agent_files = [f for f in os.listdir(log_dir) if f.startswith('Agent_') and f.endswith('.csv')]
        
        if not agent_files:
            print(f"No agent data files found in {log_dir}")
            # 기본값으로 결과 반환
            return {
                'total_distance': 0.0,
                'collision_count': 0,
                'min_obstacle_distance': float('inf'),
                'reached_goal': False,
                'operation_time': 0.0
            }
        
        # 충돌 로그 파일 찾기
        collision_file = f"{log_dir}/collision_log.csv"
        
        # 가장 최근 파일 사용
        latest_agent_file = sorted(agent_files)[-1]
        agent_data = pd.read_csv(f'{log_dir}/{latest_agent_file}')
        
        # 분석 메트릭 계산
        total_distance = 0
        min_obstacle_distance = float('inf')
        
        for i in range(1, len(agent_data)):
            # 이동 거리 계산
            dx = agent_data['x'].iloc[i] - agent_data['x'].iloc[i-1]
            dy = agent_data['y'].iloc[i] - agent_data['y'].iloc[i-1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # 충돌 정보 읽기
        collision_count = 0
        if os.path.exists(collision_file):
            collision_data = pd.read_csv(collision_file)
            collision_count = len(collision_data)  # 충돌 횟수는 로그 파일의 행 수
            if not collision_data.empty:
                min_obstacle_distance = collision_data['distance'].min()  # 최소 장애물 거리
        
        # 목표 도달 여부 확인
        final_pos = np.array([agent_data['x'].iloc[-1], agent_data['y'].iloc[-1]])
        goal_pos = np.array([17.0, 17.0])  # 목표 위치
        reached_goal = np.linalg.norm(final_pos - goal_pos) < 1.0
        
        # 운행 시간 계산 (마지막 타임스탬프)
        operation_time = agent_data['timestamp'].iloc[-1]
        
        print(f"\nAnalysis Results:")
        print(f"Total distance: {total_distance:.2f}")
        print(f"Collision count: {collision_count}")
        print(f"Minimum obstacle distance: {min_obstacle_distance:.2f}")
        print(f"Reached goal: {reached_goal}")
        print(f"Operation time: {operation_time:.2f}s")
        
        return {
            'total_distance': total_distance,
            'collision_count': collision_count,
            'min_obstacle_distance': min_obstacle_distance,
            'reached_goal': reached_goal,
            'execution_time': len(agent_data) * 0.1,
            'operation_time': operation_time
        }
    except Exception as e:
        print(f"Error in analyze_results: {e}")
        return {
            'total_distance': 0.0,
            'collision_count': 0,
            'min_obstacle_distance': float('inf'),
            'reached_goal': False,
            'operation_time': 0.0
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

def save_results_to_csv(results, algorithm, scenario_num):
    """결과를 CSV 파일에 누적하여 저장"""
    results_file = 'all_results.csv'
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    # 결과 데이터 준비
    result_row = {
        'timestamp': current_time,
        'algorithm': algorithm,
        'scenario': scenario_num,
        'total_distance': results['total_distance'],
        'collision_count': results['collision_count'],
        'min_obstacle_distance': results['min_obstacle_distance'],
        'reached_goal': results['reached_goal'],
        'execution_time': results['execution_time'],
        'operation_time': results['operation_time']  # 운행 시간 추가
    }
    
    # 파일이 존재하는지 확인
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)  # append 대신 concat 사용
    else:
        df = pd.DataFrame([result_row])
    
    # CSV 파일로 저장
    df.to_csv(results_file, index=False)

def main():
    num_iterations = 1
    algorithms = ["DDPG", "DDPG_DTP", "DWA"]
    scenario_num = 1

    print(f"Starting {num_iterations} iterations of experiments...")
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print('='*50)
        
        results = {}
        
        # 각 알고리즘 실행 및 분석
        for algorithm in algorithms:
            try:
                print(f"\nTesting {algorithm}...")
                log_dir, execution_time = run_simulation(algorithm)
                
                if log_dir is None:
                    print(f"Warning: Simulation failed for {algorithm}")
                    continue
                
                # 결과 분석
                analysis_result = analyze_results(log_dir)
                if analysis_result is None:
                    print(f"Warning: Analysis failed for {algorithm}")
                    continue
                
                # 결과 저장
                results[algorithm] = analysis_result
                results[algorithm]['execution_time'] = execution_time
                
                # 시각화
                visualize_trajectory(log_dir)
                visualize_trajectory_with_obstacles(log_dir)
                
                # CSV에 결과 저장
                save_results_to_csv(results[algorithm], algorithm, scenario_num)
                
            except Exception as e:
                print(f"Error processing {algorithm}: {str(e)}")
                continue
        
        # 결과가 있는 경우에만 비교 및 시각화 수행
        if results:
            try:
                # 현재 반복의 결과 비교 및 출력
                print(f"\nResults Comparison (Iteration {iteration + 1}):")
                print("-" * 50)
                metrics = ['total_distance', 'collision_count', 'min_obstacle_distance', 
                          'reached_goal', 'execution_time']
                
                for metric in metrics:
                    print(f"\n{metric}:")
                    for alg in results.keys():
                        if metric in results[alg]:
                            print(f"{alg}: {results[alg][metric]:.2f}")
                
                # 현재 반복의 결과를 CSV 파일로 저장
                results_df = pd.DataFrame(results).T
                results_df.to_csv(f'Result/simulation_results_iter_{iteration+1}.csv')
                
                # 현재 반복의 결과 시각화
                plt.figure(figsize=(12, 6))
                for metric in metrics[:-1]:  # execution_time 제외
                    plt.subplot(2, 2, metrics.index(metric) + 1)
                    valid_algs = [alg for alg in results.keys() if metric in results[alg]]
                    if valid_algs:
                        plt.bar(valid_algs, [results[alg][metric] for alg in valid_algs])
                        plt.title(f'{metric} (Iter {iteration+1})')
                        plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'Result/comparison_results_iter_{iteration+1}.png')
                plt.close()
                
            except Exception as e:
                print(f"Error in results processing: {str(e)}")
        else:
            print(f"No valid results for iteration {iteration + 1}")

    print("\nAll iterations completed!")
    print(f"Results saved in simulation_results_iter_*.csv files")
    print(f"Plots saved in comparison_results_iter_*.png files")

if __name__ == "__main__":
    main() 