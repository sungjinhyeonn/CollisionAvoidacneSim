import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from network import ActorCritic
from custom_nav_env import Navigation2DEnv
import time
import os

def load_and_test_model(model_path, num_episodes=10, render=True):
    # 환경 생성
    env = Navigation2DEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 파일 존재 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # 모델 생성
        model = ActorCritic(state_dim, action_dim).to(device)
        
        # 모델 로드 시도
        checkpoint = torch.load(model_path, map_location=device)
        
        # 체크포인트 구조 확인
        if 'model_state_dict' not in checkpoint:
            print("Warning: model_state_dict not found in checkpoint")
            print("Available keys:", checkpoint.keys())
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)  # 직접 state dict인 경우
            else:
                raise ValueError("Invalid checkpoint format")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        print(f"\nLoaded model from {model_path}")
        if 'reward' in checkpoint:
            print(f"Best reward achieved during training: {checkpoint['reward']:.2f}")
        if 'timestep' in checkpoint:
            print(f"Saved at timestep: {checkpoint['timestep']}")
            
        return model, env
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

if __name__ == "__main__":
    # 모델 경로 설정
    model_path = 'best_navigation_model.pth'  # 또는 절대 경로 사용
    
    try:
        # 모델 테스트 실행
        model, env = load_and_test_model(
            model_path=model_path,
            num_episodes=10,  # 테스트할 에피소드 수
            render=True      # 시각화 여부
        )
        
        print("\nStarting evaluation...")
        for episode in range(10):
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                # 행동 선택
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = model.get_action(state_tensor, deterministic=True)
                
                # 환경과 상호작용
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if render:
                    env.render()
                    time.sleep(0.01)
                
                state = next_state
                step += 1
            
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {step}")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nError occurred during testing: {str(e)}")
    finally:
        plt.close('all')  # 모든 플롯 창 닫기