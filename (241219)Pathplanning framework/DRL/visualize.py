import torch
import numpy as np
import pygame
import os
from env import Environment
import numpy as np
from env import Environment
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import os
# Actor 클래스 직접 정의 (DRL.py에서 가져옴)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def visualize_model(model_name="best_model_ENU3"):
    """특정 학습된 모델의 성능을 시각화"""
    try:
        # 모델 경로 설정
        model_path = os.path.join("DRL", f"{model_name}.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model: {model_path}")
        
        # 환경 생성
        env = Environment(num_rays=8)
        
        # Actor 모델 생성 및 가중치 로드
        state_dim = 4 + env.num_rays  # robot_position(2) + distance_to_goal(1) + angle_to_goal(1) + rays(8)
        action_dim = 2  # linear velocity, angular velocity
        
        actor = Actor(state_dim, action_dim)
        actor.load_state_dict(torch.load(model_path))
        actor.eval()
        
        print("Model loaded successfully")
        
        # Pygame 초기화
        pygame.init()
        screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption(f"Model Visualization - {model_name}")
        clock = pygame.time.Clock()
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # R키: 리셋
                        obs = env.reset()
                        total_reward = 0
                        steps = 0
                    elif event.key == pygame.K_ESCAPE:  # ESC키: 종료
                        running = False
            
            # 상태 벡터 구성
            state = np.concatenate([
                obs['robot_position'],
                [obs['distance_to_goal']],
                [obs['angle_to_goal']],
                [ray[2] for ray in obs['rays']]
            ])
            
            # 모델로부터 행동 얻기
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = actor(state_tensor).squeeze(0).cpu().numpy()
            
            # 환경 진행
            obs, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            # 화면 지우기
            screen.fill((255, 255, 255))
            
            # 환경 렌더링 (왼쪽)
            env.render_with_rays(screen.subsurface((0, 0, 400, 400)))
            
            # 정보 표시 (오른쪽)
            info_surface = screen.subsurface((400, 0, 400, 400))
            font = pygame.font.Font(None, 36)
            
            # 현재 행동 표시
            text_action = font.render(f"Linear: {action[0]:.2f}", True, (0, 0, 0))
            info_surface.blit(text_action, (20, 20))
            text_angular = font.render(f"Angular: {action[1]:.2f}", True, (0, 0, 0))
            info_surface.blit(text_angular, (20, 60))
            
            # 현재 보상 표시
            text_reward = font.render(f"Reward: {reward:.2f}", True, (0, 0, 0))
            info_surface.blit(text_reward, (20, 100))
            
            # 총 보상 표시
            text_total = font.render(f"Total: {total_reward:.2f}", True, (0, 0, 0))
            info_surface.blit(text_total, (20, 140))
            
            # 스텝 수 표시
            text_steps = font.render(f"Steps: {steps}", True, (0, 0, 0))
            info_surface.blit(text_steps, (20, 180))
            
            # 조작 방법 표시
            text_help1 = font.render("R: Reset", True, (100, 100, 100))
            text_help2 = font.render("ESC: Quit", True, (100, 100, 100))
            info_surface.blit(text_help1, (20, 300))
            info_surface.blit(text_help2, (20, 340))
            
            pygame.display.flip()
            clock.tick(30)
            
            if done:
                print(f"Episode finished after {steps} steps. Total reward: {total_reward:.2f}")
                obs = env.reset()
                total_reward = 0
                steps = 0
        
        pygame.quit()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_model("best_model_ENU3")
