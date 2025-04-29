import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle
from warehouse_layout import LAYOUT

# 전역 변수로 충돌 카운터 추가
collision_counts = {}  # 각 에이전트별 충돌 횟수
last_collision_states = {}  # 이전 프레임의 충돌 상태
collision_positions = []  # 충돌 위치를 저장할 리스트

def load_agent_data(file_directory):
    agent_data_list = []
    for file_name in os.listdir(file_directory):
        if file_name.startswith('Agent_') and file_name.endswith('.csv'):
            file_path = os.path.join(file_directory, file_name)
            agent_data = pd.read_csv(file_path)
            agent_id = file_name.replace('Agent_', '').replace('.csv', '')
            agent_data['Agent'] = agent_id
            agent_data_list.append(agent_data)
    return pd.concat(agent_data_list, ignore_index=True)

def load_obstacle_data(file_directory):
    obstacle_data_list = []
    for file_name in os.listdir(file_directory):
        if file_name.startswith('Obstacle_') and file_name.endswith('.csv'):
            file_path = os.path.join(file_directory, file_name)
            obstacle_data = pd.read_csv(file_path)
            obstacle_id = file_name.replace('Obstacle_', '').replace('.csv', '')
            obstacle_data['Obstacle'] = obstacle_id
            obstacle_data_list.append(obstacle_data)
    return pd.concat(obstacle_data_list, ignore_index=True)

# 레이아웃 정의를 LAYOUT으로 대체
layout = LAYOUT

def is_near_point(agent_pos, point, threshold=2.0):
    """에이전트가 특정 포인트 근처에 있는지 확인"""
    return np.sqrt((agent_pos['x'] - point[0])**2 + (agent_pos['y'] - point[1])**2) < threshold

def plot_layout(ax, highlight_points=None):
    # 보관 선반 그리기
    for rack in layout['storage_racks']:
        polygon = Polygon(rack, facecolor='lightgray', alpha=0.5)
        ax.add_patch(polygon)
    
    # # 출발지 구역 그리기 (안전 구역 그리기 전에 추가)
    # for zone in layout['start_zones']:
    #     polygon = Polygon(zone, facecolor='lightgreen', alpha=0.3,
    #                     edgecolor='green', linewidth=1, zorder=1)
    #     ax.add_patch(polygon)
    #     # 구역 중심에 'S' 텍스트 추가
    #     center_x = sum(p[0] for p in zone) / 4
    #     center_y = sum(p[1] for p in zone) / 4
    #     ax.text(center_x, center_y, 'S', 
    #             horizontalalignment='center',
    #             verticalalignment='center',
    #             color='darkgreen',
    #             fontsize=12,
    #             fontweight='bold')
    
    # 피킹 위치 그리기 - 색상 반전 효과 추가
    for i, point in enumerate(layout['pickup_points']):
        if highlight_points and ('P', i) in highlight_points:
            # 반전된 색상으로 큰 마커 (빨색 -> 흰색)
            ax.scatter(point[0], point[1], c='white', marker='^', s=200, 
                      edgecolor='red', linewidth=2, zorder=3)
            # 깜빡이는 효과를 위한 외곽 마커
            ax.scatter(point[0], point[1], c='none', marker='^', s=300,
                      edgecolor='red', alpha=0.5, linewidth=2, zorder=2)
        else:
            # 기본 마커
            ax.scatter(point[0], point[1], c='red', marker='^', s=100, zorder=1)
    
    # # 적재 위치 그리기 - 색상 반전 효과 추가
    # for i, point in enumerate(layout['drop_points']):
    #     if highlight_points and ('D', i) in highlight_points:
    #         # 반전된 색상으로 큰 마커 (파란색 -> 흰색)
    #         ax.scatter(point[0], point[1], c='white', marker='s', s=200,
    #                   edgecolor='blue', linewidth=2, zorder=3)
    #         # 깜빡이는 효과를 위한 외곽 마커
    #         ax.scatter(point[0], point[1], c='none', marker='s', s=300,
    #                   edgecolor='blue', alpha=0.5, linewidth=2, zorder=2)
    #     else:
    #         # 기본 마커
    #         ax.scatter(point[0], point[1], c='blue', marker='s', s=100, zorder=1)
    
    # # 안전 구역 그리기
    # for zone in layout['safety_zones']:
    #     polygon = Polygon(zone, facecolor='yellow', alpha=0.3)
    #     ax.add_patch(polygon)
    
    # 충전 스테이션 그리기
    # ax.scatter(layout['charging_station'][0], layout['charging_station'][1], 
    #           c='green', marker='*', s=150, label='Charging Station')
    
    # 레이블 추가
    for i, point in enumerate(layout['pickup_points']):
        ax.annotate(f'P{i+1}', (point[0], point[1]), xytext=(5, 5), 
                   textcoords='offset points')
    
    # for i, point in enumerate(layout['drop_points']):
    #     ax.annotate(f'D{i+1}', (point[0], point[1]), xytext=(5, 5), 
    #                textcoords='offset points')

def is_collision(robot_pos, obstacle_pos, collision_threshold=1.0):
    """로봇과 장애물 간의 충돌 검사"""
    distance = np.sqrt((robot_pos['x'] - obstacle_pos['x'])**2 + 
                      (robot_pos['y'] - obstacle_pos['y'])**2)
    return distance < collision_threshold

def draw_robot(ax, x, y, yaw, agent_id, is_colliding=False, length=1.0):
    """로봇 그리기 함수 - 충돌 상태에 따라 색상 변경"""
    robot_width = 1.5
    points = create_rectangle_points(x, y, length, robot_width, yaw)
    
    if is_colliding:
        # 충돌 시 색상 반전
        robot = Polygon(points, facecolor='red', edgecolor='white', 
                       linewidth=2, zorder=10)
    else:
        # 정상 상태
        robot = Polygon(points, facecolor='#FFA500', edgecolor='#8B4513', 
                       linewidth=2, zorder=10)
    
    ax.add_patch(robot)
    
    # 방향 표시 화살표
    dx = length/2 * np.cos(yaw)
    dy = length/2 * np.sin(yaw)
    
    if is_colliding:
        # 충돌 시 화살표 색상도 반전
        ax.arrow(x, y, dx*0.8, dy*0.8, 
                head_width=0.4, head_length=0.5,
                fc='white', ec='red',
                linewidth=2, zorder=11)
    else:
        ax.arrow(x, y, dx*0.8, dy*0.8, 
                head_width=0.4, head_length=0.5,
                fc='#CD853F', ec='#8B4513',
                linewidth=2, zorder=11)

def update(frame):
    global collision_counts, last_collision_states, collision_positions
    ax.clear()
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 40)
    
    # 배경색 설정
    ax.set_facecolor('#E0E0E0')
    fig.patch.set_facecolor('#E0E0E0')
    
    # 첫 번째 프레임에서 충돌 결과 출력
    if frame == 0:
        print("Initial Collision Counts:")
        for agent_id in sorted(collision_counts.keys()):
            print(f"Agent {agent_id}: {collision_counts[agent_id]}")
        print(f"Total: {sum(collision_counts.values())}\n")
    
    # 진행률 표시
    progress = (frame + 1) / len(times) * 100
    print(f"\rProgress: {progress:.1f}%", end="")
    
    highlight_points = set()
    current_time = times[frame]
    trail_length = 50

    # 데저 모든 에이전트의 현재 위치를 확인하여 하이라이트 포인트 결정
    for agent_id in agent_data['Agent'].unique():
        agent_data_filtered = agent_data[agent_data['Agent'] == agent_id]
        if not agent_data_filtered.empty:
            current_data = agent_data_filtered[agent_data_filtered['timestamp'] <= current_time]
            if not current_data.empty:
                current_pos = current_data.iloc[-1]
                
                # 픽업/드롭 포인트 체크
                for i, point in enumerate(layout['pickup_points']):
                    if is_near_point(current_pos, point):
                        highlight_points.add(('P', i))
                
                # for i, point in enumerate(layout['drop_points']):
                #     if is_near_point(current_pos, point):
                #         highlight_points.add(('D', i))

    # 레이아웃 그리기 (하이라이트 포인트 정보 포함)
    plot_layout(ax, highlight_points)

    # 장애물 경로 그리기 (중간 레이어)
    for obstacle_id in obstacle_data['Obstacle'].unique():
        obstacle_data_filtered = obstacle_data[obstacle_data['Obstacle'] == obstacle_id]
        if not obstacle_data_filtered.empty:
            current_idx = obstacle_data_filtered[obstacle_data_filtered['timestamp'] <= current_time].index[-1]
            
            # 현재 위치의 작업자만 그리기 (궤적 제거)
            current_pos = obstacle_data_filtered.loc[current_idx]
            worker = Circle((current_pos['x'], current_pos['y']), 
                          radius=0.5, 
                          facecolor='none', 
                          edgecolor='blue', 
                          linewidth=1.5, 
                          alpha=0.8,
                          zorder=6)
            ax.add_patch(worker)

    # 현재 시간의 장애물 위치 저장
    current_obstacle_positions = {}
    for obstacle_id in obstacle_data['Obstacle'].unique():
        obstacle_data_filtered = obstacle_data[obstacle_data['Obstacle'] == obstacle_id]
        if not obstacle_data_filtered.empty:
            current_data = obstacle_data_filtered[obstacle_data_filtered['timestamp'] <= current_time]
            if not current_data.empty:
                current_obstacle_positions[obstacle_id] = current_data.iloc[-1]

    # 에이전트 데이터 처리
    for agent_id in agent_data['Agent'].unique():
        if agent_id not in collision_counts:
            collision_counts[agent_id] = 0
            last_collision_states[agent_id] = False
            
        agent_data_filtered = agent_data[agent_data['Agent'] == agent_id]
        if not agent_data_filtered.empty:
            current_data = agent_data_filtered[agent_data_filtered['timestamp'] <= current_time]
            if not current_data.empty:
                current_idx = current_data.index[-1]
                start_idx = max(current_idx - trail_length, agent_data_filtered.index[0])
                past_data = agent_data_filtered.loc[start_idx:current_idx]
                
                if not past_data.empty:
                    # 경로 그리기
                    ax.plot(past_data['x'], past_data['y'], '-', 
                            color='#32CD32', linewidth=1, alpha=0.5, zorder=8)
                    
                    # 현재 위치에서 충돌 검사
                    current_pos = past_data.iloc[-1]
                    is_colliding = False
                    
                    # 모든 장애물과의 충돌 검사
                    for obstacle_pos in current_obstacle_positions.values():
                        if is_collision(current_pos, obstacle_pos):
                            is_colliding = True
                            # 새로운 충돌 발생 시 위치 저장
                            if not last_collision_states[agent_id]:
                                collision_positions.append((current_pos['x'], current_pos['y']))
                            break
                    
                    # 새로운 충돌 발생 시 카운트 증가
                    if is_colliding and not last_collision_states[agent_id]:
                        collision_counts[agent_id] += 1
                    
                    # 현재 충돌 상태 저장
                    last_collision_states[agent_id] = is_colliding
                    
                    # 충돌 상태를 반영하여 로봇 그리기
                    draw_robot(ax, current_pos['x'], current_pos['y'], 
                             current_pos['yaw'], agent_id, is_colliding)

    # 모든 충돌 위치에 X 표시
    for collision_pos in collision_positions:
        ax.plot(collision_pos[0], collision_pos[1], 'rx', 
                markersize=10, markeredgewidth=2, zorder=12)

    # 충돌 카운트 표시
    collision_text = "Collision Counts:\n"
    total_collisions = 0
    for agent_id in sorted(collision_counts.keys()):
        collision_text += f"Agent {agent_id}: {collision_counts[agent_id]}\n"
        total_collisions += collision_counts[agent_id]
    collision_text += f"\nTotal: {total_collisions}"
    
    # 텍스트 박스 추가
    ax.text(1.02, 0.98, collision_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            verticalalignment='top',
            fontsize=10)

    plt.tight_layout()
    return []

def create_rectangle_points(x, y, length, width, yaw=0.0):
    local_points = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])
    
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    
    rotated_points = np.dot(local_points, rotation.T)
    global_points = rotated_points + np.array([x, y])
    
    return global_points

if __name__ == "__main__":
    file_directory = 'log/scenario_16/iter_5/SAC'
    agent_data = load_agent_data(file_directory)
    obstacle_data = load_obstacle_data(file_directory)
    
    # 충돌 카운터와 위치 초기화
    collision_counts = {}
    last_collision_states = {}
    collision_positions = []
    
    # 피규어 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # 시간 데이터 준비
    all_timestamps = pd.concat([agent_data['timestamp'], 
                              obstacle_data['timestamp']])
    times = sorted(pd.unique(all_timestamps))
    
    if len(times) == 0:
        print("Error: No timestamp data found!")
        exit()
        
    print(f"Data loaded - Time range: {min(times)} to {max(times)}")
    print(f"Number of agents: {len(agent_data['Agent'].unique())}")
    print(f"Number of obstacles: {len(obstacle_data['Obstacle'].unique())}")
    
    # 프레임 수를 줄이기 위해 타임스텝을 건너뛰기
    skip_frames = 10
    times = times[::skip_frames]
    
    print("Starting visualization...")
    print(f"Total frames: {len(times)}")
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(fig, update, 
                                 frames=len(times),
                                 interval=10,
                                 blit=False,
                                 repeat=False)
    
    # 애니메이션 유지를 위한 설정
    plt.show(block=True)
    try:
        plt.show()
    except:
        pass

    # 메모리에서 애니메이션 객체가 해제되지 않도록 유지
    plt.gcf().canvas.draw_idle()
    plt.gcf().canvas.start_event_loop(0)