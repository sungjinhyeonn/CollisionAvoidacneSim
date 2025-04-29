import numpy as np
import matplotlib.pyplot as plt
import pygame

class Environment:
    def __init__(self, grid_size=(20, 20), num_dynamic_obstacles=5, num_rays=8, max_ray_length=5):
        self.grid_size = grid_size
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.num_rays = num_rays
        self.max_ray_length = max_ray_length
        self.action_history = []
        
        # 커리큘럼 학습 관련 변수 수정
        self.max_difficulty = 5
        self.current_difficulty = 0
        self.reward_threshold = 50  # 난이도 상승을 위한 평균 리워드 임계값
        self.recent_rewards = []  # 최근 리워드 저장
        self.window_size = 10  # 평균 계산을 위한 윈도우 크기
        
        self.reset()

    def get_goal_position(self):
        """커리큘럼에 따른 목표 위치 설정"""
        # 현재 난이도에 따른 최대 거리 계산 (0~1 사이 값)
        max_distance_ratio = (self.current_difficulty + 1) / self.max_difficulty
        
        # 최대 가능 거리 계산 (그리드 크기보다 작게 제한)
        max_x = min(int(self.grid_size[0] * max_distance_ratio), self.grid_size[0] - 1)
        max_y = min(int(self.grid_size[1] * max_distance_ratio), self.grid_size[1] - 1)
        
        # 최소 거리 설정 (시작점에서 너무 가깝지 않도록)
        min_x = max(2, int(max_x * 0.3))
        min_y = max(2, int(max_y * 0.3))
        
        # 랜덤하게 목표 위치 선택
        goal_x = np.random.randint(min_x, max_x)
        goal_y = np.random.randint(min_y, max_y)
        
        return (goal_x, goal_y)

    def update_difficulty(self, episode_reward):
        """리워드 기반 난이도 조정"""
        self.recent_rewards.append(episode_reward)
        
        # 최근 window_size 개의 에피소드 리워드만 유지
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        
        # 충분한 데이터가 쌓였을 때 평균 계산
        if len(self.recent_rewards) == self.window_size:
            avg_reward = np.mean(self.recent_rewards)
            
            # 평균 리워드가 임계값을 넘으면 난이도 상승
            if avg_reward > self.reward_threshold and self.current_difficulty < self.max_difficulty:
                self.current_difficulty += 1
                self.reward_threshold *= 1.2  # 다음 난이도의 임계값은 20% 증가
                print(f"\nDifficulty increased to {self.current_difficulty + 1}/{self.max_difficulty}")
                print(f"New reward threshold: {self.reward_threshold:.1f}")
                self.recent_rewards = []  # 리워드 히스토리 초기화

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.start = (0, 0)
        self.goal = self.get_goal_position()
        self.grid[self.start] = 2
        self.grid[self.goal] = 3
        self.robot_pos = np.array(self.start, dtype=float)
        self.robot_velocity = 0.0
        self.robot_angle = 0.0
        self.dynamic_obstacles = self.initialize_dynamic_obstacles()
        self.action_history = []
        
        print(f"Current Difficulty: {self.current_difficulty + 1}/{self.max_difficulty}, "
              f"Goal Position: {self.goal}, "
              f"Avg Reward Threshold: {self.reward_threshold:.1f}")
        
        return self.get_observation()

    def initialize_dynamic_obstacles(self):
        obstacles = []
        for _ in range(self.num_dynamic_obstacles):
            while True:
                pos = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
                if pos != self.start and pos != self.goal:
                    obstacles.append({"position": np.array(pos, dtype=float), "velocity": np.random.uniform(-0.1, 0.1, size=2)})
                    break
        return obstacles

    def update_dynamic_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            obstacle["position"] += obstacle["velocity"]
            obstacle["position"] = np.clip(obstacle["position"], 0, self.grid_size[0] - 1)

            # Randomly change velocity to simulate dynamic behavior
            if np.random.rand() < 0.1:  # 10% chance to change direction
                obstacle["velocity"] = np.random.uniform(-0.1, 0.1, size=2)

    def cast_rays(self):
        rays = []
        for i in range(self.num_rays):
            angle = self.robot_angle + i * (2 * np.pi / self.num_rays)
            detected = False
            for length in range(1, self.max_ray_length + 1):
                ray_x = self.robot_pos[0] + length * np.cos(angle)
                ray_y = self.robot_pos[1] + length * np.sin(angle)

                # Check if ray is out of bounds
                if not (0 <= ray_x < self.grid_size[0] and 0 <= ray_y < self.grid_size[1]):
                    rays.append((ray_x, ray_y, length))
                    detected = True
                    break

                # Check for dynamic obstacle collision
                for obstacle in self.dynamic_obstacles:
                    if np.linalg.norm(np.array([ray_x, ray_y]) - obstacle["position"]) < 1:
                        rays.append((ray_x, ray_y, length))
                        detected = True
                        break
                if detected:
                    break
            if not detected:
                ray_x = self.robot_pos[0] + self.max_ray_length * np.cos(angle)
                ray_y = self.robot_pos[1] + self.max_ray_length * np.sin(angle)
                rays.append((ray_x, ray_y, self.max_ray_length))

        return rays

    def get_observation(self):
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
            
            obstacles = [{"type": "dynamic", "position": tuple(obstacle["position"])} 
                        for obstacle in self.dynamic_obstacles]
            rays = self.cast_rays()
            
            observation = {
                "robot_position": tuple(self.robot_pos),
                "robot_velocity": self.robot_velocity,
                "robot_angle": self.robot_angle,
                "goal": {"type": "goal", "position": self.goal},
                "distance_to_goal": distance_to_goal,  # 목적지까지 거리 추가
                "angle_to_goal": angle_to_goal,        # 목적지까지 각도 추가
                "obstacles": obstacles,
                "rays": rays
            }
            return observation
        except Exception as e:
            print("Error in get_observation:", e)
            return None

    def step(self, action):
        # 액션 히스토리에 저장
        self.action_history.append(action)
        
        max_linear_velocity = 0.5
        max_angular_velocity = np.pi/6  # 30도로 제한 (π/6 라디안)

        linear_velocity = np.clip(action[0], -max_linear_velocity, max_linear_velocity)
        angular_velocity = np.clip(action[1], -max_angular_velocity, max_angular_velocity)

        self.robot_angle += angular_velocity
        self.robot_angle %= 2 * np.pi

        dx = linear_velocity * np.cos(self.robot_angle)
        dy = linear_velocity * np.sin(self.robot_angle)
        new_pos = self.robot_pos + np.array([dx, dy])

        # 벽과의 충돌 체크
        wall_collision = not (0 <= new_pos[0] < self.grid_size[0] - 1 and 0 <= new_pos[1] < self.grid_size[1] - 1)
        
        # 장애물과의 충돌 체크
        obstacle_collision = any(np.linalg.norm(new_pos - obstacle["position"]) < 1 for obstacle in self.dynamic_obstacles)
        
        # 충돌 여부
        collision = wall_collision or obstacle_collision

        if not collision:
            self.robot_pos = new_pos
            self.robot_velocity = linear_velocity
        else:
            self.robot_velocity = 0.0

        self.update_dynamic_obstacles()
        reward = self.compute_reward(collision)
        
        # 목표 도달 또는 충돌 시 에피소드 종료
        done = tuple(self.robot_pos.astype(int)) == self.goal or collision

        if done:
            # 에피소드가 끝날 때 난이도 업데이트
            self.update_difficulty(reward)
            
        return self.get_observation(), reward, done

    def compute_reward(self, collision):
        if tuple(self.robot_pos.astype(int)) == self.goal:
            # 난이도에 따른 보상 스케일링
            return 100 * (1 + self.current_difficulty * 0.2)  # 난이도가 높을수록 더 큰 보상
        elif collision:
            return -50  # 충돌 페널티
        else:
            distance_to_goal = np.linalg.norm(self.robot_pos - np.array(self.goal))
            time_penalty = -0.3  # 시간 페널티 증가
            distance_penalty = -0.2 * distance_to_goal  # 거리 페널티 증가
            
            # 목표를 향해 전진하는 방향으로 움직일 때 약간의 보상
            goal_angle = np.arctan2(self.goal[1] - self.robot_pos[1], 
                                   self.goal[0] - self.robot_pos[0])
            angle_diff = abs(goal_angle - self.robot_angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # 각도 차이를 [0, π] 범위로
            direction_reward = 0.2 * (1 - angle_diff/np.pi)  # 올바른 방향을 향할 때 보상
            
            return time_penalty + distance_penalty + direction_reward

    def render(self):
        # Pygame 초기화
        pygame.init()
        screen = pygame.display.set_mode((400, 400))  # 화면 크기 설정
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((255, 255, 255))  # 배경색 설정 (흰색)

            # 현재 상태를 화면에 그리기
            pygame.draw.circle(screen, (0, 0, 255), 
                             (int(self.robot_pos[1] * 40), int(self.robot_pos[0] * 40)), 10)

            # 차량의 헤딩 표시 (붉은색 삼각형)
            heading_angle = np.arctan2(self.robot_pos[1], self.robot_pos[0])
            triangle_points = [
                (int(self.robot_pos[1] * 40), int(self.robot_pos[0] * 40)),
                (int((self.robot_pos[1] + 0.5 * np.cos(heading_angle)) * 40), 
                 int((self.robot_pos[0] + 0.5 * np.sin(heading_angle)) * 40)),
                (int((self.robot_pos[1] + 0.3 * np.cos(heading_angle + np.pi / 6)) * 40), 
                 int((self.robot_pos[0] + 0.3 * np.sin(heading_angle + np.pi / 6)) * 40)),
                (int((self.robot_pos[1] + 0.3 * np.cos(heading_angle - np.pi / 6)) * 40), 
                 int((self.robot_pos[0] + 0.3 * np.sin(heading_angle - np.pi / 6)) * 40))
            ]
            pygame.draw.polygon(screen, (255, 0, 0), triangle_points)

            # 레이캐스트 가시화
            rays = self.cast_rays()
            for ray in rays:
                pygame.draw.line(screen, (255, 0, 0), 
                               (int(self.robot_pos[1] * 40), int(self.robot_pos[0] * 40)), 
                               (int(ray[0] * 40), int(ray[1] * 40)), 1)

            # 장애물 그리기 - 수정된 부분
            for obstacle in self.dynamic_obstacles:
                obstacle_pos = obstacle["position"]  # 장애물 위치 벡터
                # 레이와 장애물 간의 충돌 검사
                color = (255, 0, 0) if any(np.linalg.norm(obstacle_pos - np.array([ray[0], ray[1]])) < 0.5 
                                         for ray in rays) else (255, 255, 0)
                pygame.draw.circle(screen, color, 
                                 (int(obstacle_pos[0] * 40), int(obstacle_pos[1] * 40)), 10)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def render_with_rays(self, screen):
        grid_size = 20

        # Draw the environment grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                color = (200, 200, 200)
                if (x, y) == self.goal:
                    color = (255, 0, 0)  # Goal
                pygame.draw.rect(screen, color, pygame.Rect(y * grid_size, x * grid_size, grid_size, grid_size))

        # Draw the robot
        robot_x, robot_y = self.robot_pos
        pygame.draw.circle(screen, (0, 0, 255), (int(robot_y * grid_size + grid_size / 2), int(robot_x * grid_size + grid_size / 2)), grid_size // 4)

        # Draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            ox, oy = obstacle["position"]
            pygame.draw.circle(screen, (255, 255, 0), (int(oy * grid_size + grid_size / 2), int(ox * grid_size + grid_size / 2)), grid_size // 4)

        # Draw rays
        rays = self.cast_rays()
        for ray in rays:
            ray_x, ray_y, _ = ray
            pygame.draw.line(screen, (0, 255, 0),
                             (int(self.robot_pos[1] * grid_size + grid_size / 2), int(self.robot_pos[0] * grid_size + grid_size / 2)),
                             (int(ray_y * grid_size + grid_size / 2), int(ray_x * grid_size + grid_size / 2)), 1)

    def render_with_rays_and_action(self, screen, action=None):
        # 기존의 환경 렌더링
        self.render_with_rays(screen)
        
        # 액션 히스토리 그래프 그리기 (화면 우측에)
        if len(self.action_history) > 0:
            graph_width = 150
            graph_height = 200
            graph_x = 400 - graph_width - 10  # 우측 여백
            graph_y = 10  # 상단 여백
            
            # 그래프 배경
            pygame.draw.rect(screen, (240, 240, 240), 
                           (graph_x, graph_y, graph_width, graph_height))
            
            # 데이터 포인트 변환
            max_points = 50  # 표시할 최대 데이터 포인트 수
            history = self.action_history[-max_points:] if len(self.action_history) > max_points else self.action_history
            
            # 선속도 그래프 (파란색)
            points_v = [(graph_x + i * (graph_width / len(history)), 
                        graph_y + graph_height/4 * (1 - h[0]))
                       for i, h in enumerate(history)]
            if len(points_v) > 1:
                pygame.draw.lines(screen, (0, 0, 255), False, points_v, 2)
            
            # 각속도 그래프 (빨간색)
            points_w = [(graph_x + i * (graph_width / len(history)), 
                        graph_y + graph_height/4 * 3 * (1 - h[1]/np.pi))
                       for i, h in enumerate(history)]
            if len(points_w) > 1:
                pygame.draw.lines(screen, (255, 0, 0), False, points_w, 2)
            
            # 레이블 표시
            font = pygame.font.Font(None, 24)
            v_text = font.render('Linear Vel', True, (0, 0, 255))
            w_text = font.render('Angular Vel', True, (255, 0, 0))
            screen.blit(v_text, (graph_x, graph_y + graph_height + 5))
            screen.blit(w_text, (graph_x, graph_y + graph_height + 25))
            
            # 현재 값 표시
            if action is not None:
                current_v = font.render(f'{action[0]:.2f}', True, (0, 0, 255))
                current_w = font.render(f'{action[1]:.2f}', True, (255, 0, 0))
                screen.blit(current_v, (graph_x + graph_width - 50, graph_y + graph_height + 5))
                screen.blit(current_w, (graph_x + graph_width - 50, graph_y + graph_height + 25))

    def visualize_action(self, action):
        """
        액션(선속도, 각속도)을 시각화하는 함수
        """
        plt.figure(figsize=(4, 4))
        
        # 원형 게이지 그리기
        circle = plt.Circle((0, 0), 1.0, fill=False, color='gray')
        plt.gca().add_patch(circle)
        
        # 선속도를 화살표 길이로, 각속도를 화살표 방향으로 표시
        linear_vel, angular_vel = action
        
        # 선속도와 각속도를 -1~1 범위로 정규화
        linear_vel = np.clip(linear_vel, -1, 1)
        angular_vel = np.clip(angular_vel, -1, 1)
        
        # 화살표 각도 계산 (각속도에 따라 회전)
        angle = np.pi/2 + angular_vel * np.pi/2
        
        # 화살표 길이는 선속도에 비례
        arrow_length = abs(linear_vel)
        
        # 화살표 색상 (전진: 파란색, 후진: 빨간색)
        color = 'blue' if linear_vel >= 0 else 'red'
        
        # 화살표 그리기
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        plt.arrow(0, 0, dx, dy, 
                 head_width=0.1, 
                 head_length=0.2, 
                 fc=color, 
                 ec=color,
                 alpha=0.7)
        
        # 축 설정
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.gca().set_aspect('equal')
        
        # 격자 그리기
        plt.grid(True)
        
        # 제목 설정
        plt.title(f'Linear Vel: {linear_vel:.2f}, Angular Vel: {angular_vel:.2f}')
        
        # 작은 창으로 표시
        plt.tight_layout()
        
        # 별도의 창에 표시
        plt.show(block=False)
        plt.pause(0.01)

    def visualize_episode_actions(self):
        """
        에피소드가 끝난 후 전체 액션 히스토리를 한 번에 시각화
        """
        if len(self.action_history) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        
        # 시간 축 생성
        time_steps = np.arange(len(self.action_history))
        
        # 선속도와 각속도 분리
        linear_vel = [action[0] for action in self.action_history]
        angular_vel = [action[1] for action in self.action_history]
        
        # 서브플롯 생성
        plt.subplot(2, 1, 1)
        plt.plot(time_steps, linear_vel, 'b-', label='Linear Velocity')
        plt.grid(True)
        plt.ylabel('Linear Velocity')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, angular_vel, 'r-', label='Angular Velocity')
        plt.grid(True)
        plt.xlabel('Time Steps')
        plt.ylabel('Angular Velocity')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Pygame setup for manual control
# Pygame setup for manual control
def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    env = Environment(num_rays=16)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = [0.0, 0.0]  # [linear_velocity, angular_velocity]

        # Keyboard controls
        if keys[pygame.K_UP]:
            action[0] += 0.1  # Increase linear velocity
        if keys[pygame.K_DOWN]:
            action[0] -= 0.1  # Decrease linear velocity
        if keys[pygame.K_LEFT]:
            action[1] -= 0.1  # Turn left
        if keys[pygame.K_RIGHT]:
            action[1] += 0.1  # Turn right

        # Update environment
        observation, reward, done = env.step(action)

        # Check if observation is None
        if observation is None:
            print("Observation is None, resetting environment.")
            observation = env.reset()

        # Render environment, rays, and action visualization
        screen.fill((255, 255, 255))
        env.render_with_rays_and_action(screen, action)  # 액션 시각화 추가
        pygame.display.flip()

        clock.tick(30)

        if done:
            print("Goal reached!")
            env.reset()

    pygame.quit()

if __name__ == "__main__":
    main()
