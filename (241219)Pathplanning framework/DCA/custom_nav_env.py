import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class Navigation2DEnv(gym.Env):
    def __init__(self):
        super(Navigation2DEnv, self).__init__()
        
        # 환경 설정
        self.size = 40  # 환경 크기
        self.robot_radius = 1.0
        self.goal_radius = 1.0
        self.obstacle_radius = 2.0
        
        # 장애물 관련 설정
        self.max_obstacles = 5  # 최대 장애물 수
        self.num_obstacles = 2  # 현재 장애물 수 (초기값)
        
        # 로봇 동역학 파라미터 추가
        self.dt = 0.1  # 시간 간격
        self.max_velocity = 2.0  # 최대 속도
        self.max_acceleration = 1.0  # 최대 가속도
        self.velocity_decay = 0.95  # 속도 감쇠율
        
        # 로봇 상태 확장 (위치 + 속도)
        self.robot_vel = np.zeros(2)  # 로봇의 현재 속도
        
        # 장애물 관련 설정
        self.min_obstacle_velocity = -0.2
        self.max_obstacle_velocity = 0.2
        self.max_goal_distance = 20
        
        # 상태 공간 차원 계산
        self.state_dim = 6 + 4 * self.max_obstacles  # 로봇위치(2) + 로봇속도(2) + 목표위치(2) + 장애물정보(4*max_obstacles)
        print(f"Fixed state dimension: {self.state_dim}")
        
        # 액션 및 관찰 공간 정의
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        # 로봇 초기화
        self.robot_pos = np.random.uniform(5, self.size-5, 2)
        self.robot_vel = np.zeros(2)  # 초기 속도는 0
        
        # 목표 위치 설정
        while True:
            self.goal_pos = np.random.uniform(5, self.size-5, 2)
            if np.linalg.norm(self.goal_pos - self.robot_pos) <= self.max_goal_distance:
                break
        
        # 장애물 초기화
        self.obstacles = []
        self.obstacle_velocities = []
        
        while len(self.obstacles) < self.num_obstacles:
            obs_pos = np.random.uniform(5, self.size-5, 2)
            if (np.linalg.norm(obs_pos - self.robot_pos) > 5 and 
                np.linalg.norm(obs_pos - self.goal_pos) > 5):
                self.obstacles.append(obs_pos)
                velocity = np.random.uniform(
                    self.min_obstacle_velocity,
                    self.max_obstacle_velocity,
                    2
                )
                self.obstacle_velocities.append(velocity)
        
        self.obstacles = np.array(self.obstacles)
        self.obstacle_velocities = np.array(self.obstacle_velocities)
        
        return self._get_obs()
    
    def _get_obs(self):
        """상태 관측값 반환"""
        # 로봇과 목표 상대 위치
        robot_goal_vec = self.goal_pos - self.robot_pos
        
        # 장애물 관련 정보
        obstacle_info = []
        for i in range(self.max_obstacles):
            if i < len(self.obstacles):
                obstacle_vec = self.obstacles[i] - self.robot_pos
                velocity = self.obstacle_velocities[i]
                obstacle_info.extend([*obstacle_vec, *velocity])
            else:
                obstacle_info.extend([0, 0, 0, 0])
        
        # 모든 정보를 하나의 벡터로 연결
        state = np.concatenate([
            robot_goal_vec,  # 2
            self.robot_pos,  # 2
            self.robot_vel,  # 2 (속도 정보 추가)
            np.array(obstacle_info)  # 4 * max_obstacles
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        # 이전 상태 저장
        prev_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        # 가속도 기반 제어
        acceleration = np.clip(action, -self.max_acceleration, self.max_acceleration)
        
        # 속도 업데이트 (관성 및 감쇠 고려)
        self.robot_vel = self.robot_vel * self.velocity_decay + acceleration * self.dt
        
        # 최대 속도 제한
        velocity_magnitude = np.linalg.norm(self.robot_vel)
        if velocity_magnitude > self.max_velocity:
            self.robot_vel = self.robot_vel * self.max_velocity / velocity_magnitude
        
        # 위치 업데이트
        self.robot_pos += self.robot_vel * self.dt
        
        # 경계 처리
        self.robot_pos = np.clip(self.robot_pos, 0, self.size)
        
        # 벽과 충돌 시 속도 감소
        if np.any(self.robot_pos <= 0) or np.any(self.robot_pos >= self.size):
            self.robot_vel *= 0.5
        
        # 장애물 업데이트
        self.obstacles += self.obstacle_velocities
        self.obstacles = np.clip(self.obstacles, 0, self.size)
        wall_collision = (self.obstacles <= 0) | (self.obstacles >= self.size)
        self.obstacle_velocities[wall_collision] *= -1
        
        # 보상 계산
        current_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = (prev_distance - current_distance) * 10
        
        # 목표 도달 판정
        if current_distance < self.goal_radius:
            goal_reward = 100 * (1 - current_distance/self.goal_radius)
            reward += goal_reward
            if current_distance < self.goal_radius * 0.5:
                done = True
                reward = 100
                return self._get_obs(), reward, done, {}
        
        # 충돌 판정
        for obs in self.obstacles:
            obstacle_distance = np.linalg.norm(self.robot_pos - obs)
            if obstacle_distance < self.obstacle_radius + self.robot_radius:
                collision_penalty = -100 * (1 - obstacle_distance/(self.obstacle_radius + self.robot_radius))
                reward += collision_penalty
                if obstacle_distance < (self.obstacle_radius + self.robot_radius) * 0.8:
                    done = True
                    reward = -100
                    return self._get_obs(), reward, done, {}
        
        # 에너지 효율성 고려
        reward -= 0.1 * np.linalg.norm(acceleration)  # 급격한 가속에 대한 페널티
        
        done = False
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            plt.clf()
            plt.xlim(0, self.size)
            plt.ylim(0, self.size)
            
            # 로봇 위치
            plt.plot(self.robot_pos[0], self.robot_pos[1], 'bo', markersize=10, label='Robot')
            
            # 목표 위치
            plt.plot(self.goal_pos[0], self.goal_pos[1], 'g*', markersize=15, label='Goal')
            
            # 장애물
            for obs in self.obstacles:
                plt.plot(obs[0], obs[1], 'rx', markersize=10)
            
            plt.legend()
            plt.draw()
            plt.pause(0.01)
        
        pass 

    def update_difficulty(self, num_obstacles, max_velocity, max_goal_distance):
        """난이도 파라미터 업데이트"""
        self.num_obstacles = num_obstacles
        self.max_velocity = max_velocity
        self.max_goal_distance = max_goal_distance
        
        # 상태 공간 크기 업데이트
        obs_dim = 6 + 4 * self.num_obstacles  # 로봇위치(2) + 로봇속도(2) + 목표위치(2) + 장애물정보(4*num_obstacles)
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'),
            shape=(obs_dim,),  # 로봇-목표 상대위치(2) + 로봇위치(2) + 장애물정보(4*num_obstacles)
            dtype=np.float32
        )
        
        pass 