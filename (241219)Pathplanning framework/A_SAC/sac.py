import numpy as np
from sac_env import Environment
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import os
import time

class TrainingCallback(BaseCallback):
    def __init__(self, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Pygame 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("SAC Training Visualization")
        self.clock = pygame.time.Clock()

    def _on_step(self):
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # 환경 렌더링
        self.training_env.envs[0].render_with_rays(self.screen)
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS로 제한
        
        # 현재 에피소드의 리워드 누적
        info = self.locals.get('info')
        if info is not None:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                if len(self.episode_rewards) > 30:
                    self.episode_rewards.pop(0)
        
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                std_reward = np.std(self.episode_rewards)
                max_reward = np.max(self.episode_rewards)
                min_reward = np.min(self.episode_rewards)
                
                print(f"\nStep: {self.n_calls}")
                print(f"Recent Episodes Stats:")
                print(f"  Mean reward: {mean_reward:.2f}")
                print(f"  Std reward: {std_reward:.2f}")
                print(f"  Max reward: {max_reward:.2f}")
                print(f"  Min reward: {min_reward:.2f}")
                print(f"  Current difficulty: {self.training_env.envs[0].current_difficulty}")
                print(f"  Current threshold: {self.training_env.envs[0].reward_threshold:.2f}\n")
        
        return True

    def _on_training_end(self):
        # 학습 종료 시 Pygame 종료
        pygame.quit()

def main():
    # 현재 스크립트의 디렉토리 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 현재 시간을 이용하여 고유한 실행 ID 생성
    run_id = time.strftime("%Y%m%d_%H%M%S")
    
    # tensorboard 로그 디렉토리 생성
    log_dir = os.path.join(current_dir, "sac_tensorboard", run_id)
    os.makedirs(log_dir, exist_ok=True)
    
    # 모델 저장 디렉토리 생성
    model_dir = os.path.join(current_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Log directory: {log_dir}")
    print(f"Model directory: {model_dir}")
    
    # 환경 생성 (시작 장애물 수 5개로 수정)
    env = Environment(
        grid_size=(20, 20),
        num_dynamic_obstacles=5,  # 3 → 5로 수정
        num_rays=8,
        max_ray_length=3,
        consecutive_successes_required=20
    )

    # SAC 모델 생성 (최적화된 하이퍼파라미터 적용)
    model = SAC(
        "MlpPolicy", 
        env,
        learning_rate= 0.0001135420810367477,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=96,
        tau=0.017352441918687223,
        gamma=0.9693196964031553,
        train_freq=8,
        gradient_steps=6,
        ent_coef="auto",
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        ),
        verbose=1
    )
    
    # 콜백 설정
    callback = TrainingCallback()
    
    try:
        # 학습 시작
        print("Starting training...")
        print("Using balanced hyperparameters:")
        print(f"Learning rate: {0.0024251124513035398}")
        print(f"Batch size: {224}")
        print(f"Buffer size: {10000}")
        print(f"Gamma: {0.9262384922982472}")
        print(f"Tau: {0.03691504005243593}")
        print(f"Train frequency: {2}")
        print(f"Gradient steps: {4}")
        
        model.learn(
            total_timesteps=1000000,
            callback=callback,
            log_interval=10
        )
        
        model_path = os.path.join(model_dir, f"sac_robot_{run_id}")
        model.save(model_path)
        print(f"Training completed and model saved to {model_path}!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model_path = os.path.join(model_dir, f"sac_robot_interrupted_{run_id}")
        model.save(model_path)
        print(f"Model saved to {model_path}!")
        
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()