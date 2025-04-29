import gym
import torch
from stable_baselines3 import SAC
import numpy as np
import pygame
import os

# 환경 초기화
from sac_env import Environment  # 사용자 정의 환경

def visualize_model(model_path):
    """학습된 모델의 성능을 시각화"""
    # 환경 생성
    env = Environment(num_rays=8)
    
    # 모델 불러오기
    model = SAC.load(model_path)
    
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((800, 400))  # 더 큰 화면으로 설정
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
                if event.key == pygame.K_r:  # R키를 누르면 환경 리셋
                    obs = env.reset()
                    total_reward = 0
                    steps = 0
                elif event.key == pygame.K_ESCAPE:  # ESC키를 누르면 종료
                    running = False
        
        # 모델로부터 행동 얻기
        action, _ = model.predict(obs, deterministic=True)
        
        # 환경 진행
        obs, reward, done, _ = env.step(action)
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

if __name__ == "__main__":
    try:
        # 모델 경로 설정
        models_dir = r"C:\Users\User\Desktop\CDE\(241219)Pathplanning framework\A_SAC\models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        
        if not model_files:
            raise FileNotFoundError("No trained models found!")
            
        # 가장 최근 모델 선택
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        model_path = os.path.join(models_dir, latest_model)
        print(f"Loading model: {model_path}")
        
        # 시각화 실행
        visualize_model(model_path)
        
    except Exception as e:
        print(f"Error occurred: {e}")
