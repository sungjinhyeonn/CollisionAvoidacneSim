import numpy as np
import matplotlib.pyplot as plt
import pygame
import gym
from gym import spaces
import random
class Environment(gym.Env):
    def __init__(self, grid_size=(20, 20), num_dynamic_obstacles=5, num_rays=8, max_ray_length=3,
                 initial_threshold=50, threshold_increment=20, window_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.base_obstacles = num_dynamic_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.num_rays = num_rays
        self.max_ray_length = max_ray_length
        self.action_history = []
        
        # Gym spaces 정의
        self.action_space = spaces.Box(
            low=np.array([-0.5, -np.pi/6], dtype=np.float32),
            high=np.array([0.5, np.pi/6], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space 정의
        obs_dim = 4 + num_rays
        self.observation_space = spaces.Box(
            low=np.array([-float('inf')] * obs_dim, dtype=np.float32),
            high=np.array([float('inf')] * obs_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        # 커리큘럼 학습 관련 변수 수정
        self.max_difficulty = 4
        self.current_difficulty = 0
        self.initial_threshold = initial_threshold  # 초기 임계값
        self.threshold_increment = threshold_increment  # 난이도별 증가량
        self.reward_threshold = self.initial_threshold
        self.recent_rewards = []
        self.window_size = window_size
        self.episode_rewards = []  # 에피소드 보상 기록용
        self.success_rewards = []  # 성공한 에피소드의 보상만 저장
        self.success_count = 0     # 성공 횟수 카운트
        
        self.reset()

    def seed(self, seed=None):
        """재현성(reproducibility)을 위한 랜덤 시드 고정 메서드"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]
    
    def step(self, action):
        self.action_history.append(action)
        
        max_linear_velocity = 0.5
        max_angular_velocity = np.pi/6

        linear_velocity = np.clip(action[0], -max_linear_velocity, max_linear_velocity)
        angular_velocity = np.clip(action[1], -max_angular_velocity, max_angular_velocity)

        # 로봇 각도 업데이트
        self.robot_angle += angular_velocity
        self.robot_angle %= 2 * np.pi

        # 새로운 위치 계산
        dx = linear_velocity * np.cos(self.robot_angle)
        dy = linear_velocity * np.sin(self.robot_angle)
        new_pos = self.robot_pos + np.array([dx, dy])

        # Check for collision before using it
        collision = self.check_collision()

        # 충돌이 없을 때만 위치 업데이트
        if not collision:
            self.robot_pos = new_pos
            self.robot_velocity = linear_velocity
            self.update_dynamic_obstacles()  # 장애물 위치 업데이트
        else:
            self.robot_velocity = 0.0

        observation = self.get_observation()
        if observation is None:
            # 관찰값 얻기 실패시 기본값 설정
            print("Warning: Failed to get observation in step, using default values")
            return (np.zeros(self.observation_space.shape, dtype=np.float32), 
                   -50,  # 페널티
                   True,  # 에피소드 종료
                   {})    # 추가 정보

        reward = self.compute_reward(collision)
        done = tuple(self.robot_pos.astype(int)) == self.goal or collision
        info = {}  # gym 요구사항

        if done:
            print(f"Episode Reward: {sum(self.episode_rewards):.2f}")
            print(f"Episode Result: {'Success' if tuple(self.robot_pos.astype(int)) == self.goal else 'Failed'}")
            self.update_difficulty(reward)

        # observation을 gym 형식에 맞게 변환
        obs_array = np.concatenate([
            observation['robot_position'],
            [observation['distance_to_goal']],
            [observation['angle_to_goal']],
            [ray[2] for ray in observation['rays']]
        ]).astype(np.float32)
        
        return obs_array, reward, done, info

    def reset(self):
        # 환경 초기화
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.start = (10, 10)
        
        self.goal = self.get_goal_position()
        
        # 로봇 초기 상태 설정
        self.robot_pos = np.array(self.start, dtype=float)
        self.robot_velocity = 0.0
        self.robot_angle = 0.0
        
        # 난이도에 따라 장애물 수 증가
        self.num_dynamic_obstacles = self.base_obstacles + (self.current_difficulty * 2)
        
        # 장애물 초기화
        self.dynamic_obstacles = self.initialize_dynamic_obstacles()
        self.action_history = []
        
        # 에피소드 보상 초기화 추가
        self.episode_rewards = []  # 여기서 리셋
        
        # 그리드에 시작점과 목표점 표시
        self.grid[self.start] = 2  # 시작점
        self.grid[self.goal] = 3   # 목표점
        
        # 현재 난이도 정보 출력
        print(f"\n=== New Episode ===")
        print(f"Current Difficulty: {self.current_difficulty + 1}/{self.max_difficulty}, "
              f"Goal Position: {self.goal}")
        
        # 관찰값 얻기
        observation = self.get_observation()
        if observation is None:
            # 관찰값 얻기 실패시 기본값 설정
            print("Warning: Failed to get observation, using default values")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # observation을 gym 형식에 맞게 변환
        obs_array = np.concatenate([
            observation['robot_position'],
            [observation['distance_to_goal']],
            [observation['angle_to_goal']],
            [ray[2] for ray in observation['rays']]
        ]).astype(np.float32)
        
        return obs_array

    def get_goal_position(self):
        """난이도에 따른 목표 위치 설정"""
        center = np.array([10, 10])  # 중심점
        
        # 난이도에 따른 최소/최대 거리 설정
        min_distance = 2 + self.current_difficulty * 2
        max_distance = 4 + self.current_difficulty * 2
        
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_distance, max_distance)
            
            goal = np.array([
                center[0] + distance * np.cos(angle),
                center[1] + distance * np.sin(angle)
            ]).astype(int)
            
            if (0 <= goal[0] < self.grid_size[0] and 
                0 <= goal[1] < self.grid_size[1] and 
                tuple(goal) != self.start):
                return tuple(goal)

    def check_collision(self):
        """로봇의 충돌 여부를 확인"""
        # 벽과의 충돌 체크 (여유 공간 추가)
        margin = 0.5  # 벽과의 여유 공간
        if (self.robot_pos[0] < margin or 
            self.robot_pos[0] >= self.grid_size[0] - margin or
            self.robot_pos[1] < margin or 
            self.robot_pos[1] >= self.grid_size[1] - margin):
            return True
        
        # 동적 장애물과의 충돌 체크
        for obstacle in self.dynamic_obstacles:
            if np.linalg.norm(self.robot_pos - obstacle["position"]) < 1:
                return True
                
        return False

    def get_observation(self):
        """현재 환경 상태의 관찰값 반환"""
        try:
            # 목적지까지의 거리 계산
            distance_to_goal = np.linalg.norm(self.robot_pos - np.array(self.goal))
            
            # 목적지 방향 각도 계산 (로봇 기준)
            goal_angle = np.arctan2(self.goal[1] - self.robot_pos[1], 
                                  self.goal[0] - self.robot_pos[0])
            # 로봇의 현재 방향과 목표 방향의 각도 차이
            angle_to_goal = goal_angle - self.robot_angle
            # 각도를 -π ~ π 범위로 정규화
            angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
            
            # 레이캐스트 정보 얻기
            rays = self.cast_rays()
            
            observation = {
                "robot_position": self.robot_pos,
                "robot_velocity": self.robot_velocity,
                "robot_angle": self.robot_angle,
                "distance_to_goal": distance_to_goal,
                "angle_to_goal": angle_to_goal,
                "rays": rays
            }
            return observation
        except Exception as e:
            print("Error in get_observation:", e)
            return None

    def compute_reward(self, collision):
        """보상 체계 수정"""
        # 목표 도달 또는 충돌
        if tuple(self.robot_pos.astype(int)) == self.goal:
            return 300 * (1 + self.current_difficulty * 0.5)  # 100 → 300으로 증가
        elif collision:
            return -100 * (1 + self.current_difficulty * 0.2)  # -50 → -100으로 증가
        
        # 기본 요소들
        distance_to_goal = np.linalg.norm(self.robot_pos - np.array(self.goal))
        time_penalty = -0.2  # -0.5 → -0.2로 감소
        
        # 이전 거리와 현재 거리 비교하여 페널티 계산
        prev_distance = np.linalg.norm(self.robot_pos - np.array(self.goal) - np.array([self.robot_velocity * np.cos(self.robot_angle), 
                                                                                    self.robot_velocity * np.sin(self.robot_angle)]))
        distance_diff = distance_to_goal - prev_distance
        
        # 거리가 멀어지면 더 큰 페널티
        if distance_diff > 0:
            distance_penalty = -0.3 * distance_diff  # -0.5 → -0.3으로 감소
        else:
            distance_penalty = -0.2 * distance_to_goal  # -0.3 → -0.2로 감소
        
        # 장애물 회피 보상 (감소)
        rays = self.cast_rays()
        obstacle_penalties = []
        direction_weights = [1.0] * self.num_rays
        
        for i, ray in enumerate(rays):
            distance = ray[2]
            if distance < self.max_ray_length:
                if distance < 1.0:
                    penalty = -0.5 * direction_weights[i]  # -1.0 → -0.5로 감소
                elif 2.0 < distance < 4.0:
                    penalty = 0.02 * direction_weights[i]  # 0.05 → 0.02로 감소
                else:
                    normalized_distance = distance / self.max_ray_length
                    penalty = -0.02 * direction_weights[i] * (1 - normalized_distance)  # -0.05 → -0.02로 감소
                obstacle_penalties.append(penalty)
        
        obstacle_penalty = sum(obstacle_penalties) if obstacle_penalties else 0
        
        # 움직임 관련 보상
        movement_penalty = -1.0 if abs(self.robot_velocity) < 0.1 else 0.0  # -3.0 → -1.0으로 감소
        
        # 속도 보상
        optimal_speed = 0.3
        velocity_reward = 0.2 * (1.0 - abs(self.robot_velocity - optimal_speed) / optimal_speed) \
                         if self.robot_velocity > 0 else 0  # 0.5 → 0.2로 감소
        
        # 방향 보상 (목표 지향성)
        goal_angle = np.arctan2(self.goal[1] - self.robot_pos[1], 
                               self.goal[0] - self.robot_pos[0])
        angle_diff = abs(goal_angle - self.robot_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        if angle_diff < np.pi/4:  # 전방 45도 이내
            direction_reward = 0.5  # 2.0 → 0.5로 감소
        elif angle_diff < np.pi/2:  # 전방 90도 이내
            direction_reward = 0.2  # 1.0 → 0.2로 감소
        else:
            direction_reward = -0.2  # -1.0 → -0.2로 감소
        
        # 최종 보상 계산
        total_reward = (
            time_penalty +
            distance_penalty +
            obstacle_penalty +
            movement_penalty +
            velocity_reward +
            direction_reward
        )
        
        self.episode_rewards.append(total_reward)
        return total_reward

    def get_direction_weight(self, angle):
        """각도에 따른 방향성 가중치 계산"""
        # 로봇의 현재 방향을 기준으로 상대 각도 계산
        relative_angle = (angle - self.robot_angle + np.pi) % (2 * np.pi) - np.pi
        
        # 각도별 가중치 설정
        if abs(relative_angle) <= np.pi/4:  # 전방 45도
            return 1.0
        elif abs(relative_angle) <= np.pi/2:  # 전방 90도
            return 0.8
        elif abs(relative_angle) <= 3*np.pi/4:  # 측면
            return 0.6
        elif abs(relative_angle) <= np.pi:  # 후방
            return 0.4
        return 0.2

    def get_collision_angle(self):
        """충돌이 발생한 방향의 각도 계산"""
        # 가장 가까운 장애물의 방향 계산
        closest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in self.dynamic_obstacles:
            distance = np.linalg.norm(self.robot_pos - obstacle["position"])
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle["position"]
        
        if closest_obstacle is not None:
            return np.arctan2(closest_obstacle[1] - self.robot_pos[1],
                             closest_obstacle[0] - self.robot_pos[0])
        return self.robot_angle  # 기본값으로 현재 로봇 각도 반환

    def update_difficulty(self, episode_reward):
        """난이도 업데이트 로직 수정"""
        self.recent_rewards.append(episode_reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
            
        if len(self.recent_rewards) >= self.window_size:
            avg_reward = np.mean(self.recent_rewards)
            
            if avg_reward > self.reward_threshold and self.current_difficulty < self.max_difficulty:
                self.current_difficulty += 1
                
                # 난이도별 임계값 직접 설정
                thresholds = [50, 80, 100, 110]  # 각 난이도별 임계값
                self.reward_threshold = thresholds[min(self.current_difficulty, len(thresholds)-1)]
                
                print("\n==================================================")
                print("=== Difficulty Level Up! ===")
                print(f"Current Difficulty: {self.current_difficulty + 1}/{self.max_difficulty + 1}")
                print(f"Previous Average Reward: {avg_reward:.1f}")
                print(f"New Reward Threshold: {self.reward_threshold:.1f}")
                print("==================================================\n")
                
                self.recent_rewards = []  # 보상 기록 초기화

    def initialize_dynamic_obstacles(self):
        """동적 장애물 초기화"""
        obstacles = []
        for _ in range(self.num_dynamic_obstacles):
            while True:
                pos = (
                    np.random.randint(0, self.grid_size[0]),
                    np.random.randint(0, self.grid_size[1])
                )
                # 시작점이나 목표점이 아닌 위치에 장애물 치
                if pos != self.start and pos != self.goal:
                    obstacles.append({
                        "position": np.array(pos, dtype=float),
                        "velocity": np.random.uniform(-0.1, 0.1, size=2)  # 랜덤한 초기 속도
                    })
                    break
        return obstacles

    def update_dynamic_obstacles(self):
        """동적 장애물 위치 업데이트"""
        for obstacle in self.dynamic_obstacles:
            # 현재 속도로 위치 업데이트
            obstacle["position"] += obstacle["velocity"]
            
            # 경계 체크 및 위치 보정
            obstacle["position"] = np.clip(
                obstacle["position"], 
                [0, 0], 
                [self.grid_size[0] - 1, self.grid_size[1] - 1]
            )
            
            # 랜덤하게 방향 변경 (10% 률)
            if np.random.random() < 0.1:
                obstacle["velocity"] = np.random.uniform(-0.1, 0.1, size=2)

    def cast_rays(self):
        """레이캐스트를 통한 장애물 감지"""
        rays = []
        for i in range(self.num_rays):
            angle = self.robot_angle + i * (2 * np.pi / self.num_rays)
            detected = False
            
            # 레이 길이를 1부터 최대 길이까지 증가시키며 검사
            for length in range(1, self.max_ray_length + 1):
                ray_x = self.robot_pos[0] + length * np.cos(angle)
                ray_y = self.robot_pos[1] + length * np.sin(angle)
                
                # 레이가 격자 범위를 벗어났는지 검사
                if not (0 <= ray_x < self.grid_size[0] and 0 <= ray_y < self.grid_size[1]):
                    rays.append((ray_x, ray_y, length))
                    detected = True
                    break
                
                # 동적 장애물과의 충돌 검사
                for obstacle in self.dynamic_obstacles:
                    if np.linalg.norm(np.array([ray_x, ray_y]) - obstacle["position"]) < 1:
                        rays.append((ray_x, ray_y, length))
                        detected = True
                        break
                
                if detected:
                    break
            
            # 장애물을 감지하지 못했을 경우 최대 길이의 레이 추가
            if not detected:
                ray_x = self.robot_pos[0] + self.max_ray_length * np.cos(angle)
                ray_y = self.robot_pos[1] + self.max_ray_length * np.sin(angle)
                rays.append((ray_x, ray_y, self.max_ray_length))
        
        return rays

    def render_with_rays(self, screen):
        """레이캐스트와 함께 환경을 렌더링"""
        grid_size = 20  # 각 셀의 크기

        # 배경 그리기
        screen.fill((255, 255, 255))

        # 격자 그리기
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                pygame.draw.rect(screen, (200, 200, 200), 
                               pygame.Rect(y * grid_size, x * grid_size, grid_size, grid_size), 1)

        # 목표 지점 그리기
        pygame.draw.circle(screen, (255, 0, 0),
                          (int(self.goal[1] * grid_size + grid_size/2),
                           int(self.goal[0] * grid_size + grid_size/2)), grid_size//3)

        # 동적 장애물 그리기
        for obstacle in self.dynamic_obstacles:
            pygame.draw.circle(screen, (255, 165, 0),  # 주황색
                             (int(obstacle["position"][1] * grid_size + grid_size/2),
                              int(obstacle["position"][0] * grid_size + grid_size/2)), 
                             grid_size//3)

        # 레이캐스트 그리기
        rays = self.cast_rays()
        for ray in rays:
            ray_x, ray_y, _ = ray
            pygame.draw.line(screen, (0, 255, 0),  # 초록색
                            (int(self.robot_pos[1] * grid_size + grid_size/2),
                             int(self.robot_pos[0] * grid_size + grid_size/2)),
                            (int(ray_y * grid_size + grid_size/2),
                             int(ray_x * grid_size + grid_size/2)), 1)

        # 로봇 그리기
        robot_center = (int(self.robot_pos[1] * grid_size + grid_size/2),
                       int(self.robot_pos[0] * grid_size + grid_size/2))
        
        # 로봇 본체
        pygame.draw.circle(screen, (0, 0, 255), robot_center, grid_size//3)
        
        # 로봇의 방향 표시
        direction_x = robot_center[0] + int(grid_size/2 * np.cos(self.robot_angle))
        direction_y = robot_center[1] + int(grid_size/2 * np.sin(self.robot_angle))
        pygame.draw.line(screen, (255, 0, 0), robot_center, (direction_x, direction_y), 2)

    def increase_difficulty(self):
        if self.current_difficulty < self.max_difficulty:
            print("\n==================================================")
            print("=== Difficulty Level Up! ===")
            print(f"Current Difficulty: {self.current_difficulty + 2}/{self.max_difficulty + 1}")
            print(f"Previous Success Rate: {(self.success_count / self.window_size) * 100:.1f}%")
            print(f"Previous Average Success Reward: {np.mean(self.success_rewards):.1f}")
            print(f"New Reward Threshold: {self.reward_threshold + self.threshold_increment:.1f}")
            print("==================================================\n")
            
            self.current_difficulty += 1
            self.reward_threshold += self.threshold_increment
