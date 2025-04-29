import os
import json
import optuna
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_slice,
    plot_param_importances
)
from A_SAC.rewardthreshold.sac_env_1 import Environment
import time

def evaluate_model(model, env, num_episodes=5):
    """모델 평가"""
    total_rewards = 0.0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
        total_rewards += episode_reward
    
    return total_rewards / num_episodes

def objective(trial):
    """Optuna 최적화 목적 함수"""
    # 하이퍼파라미터 설정
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    buffer_size = trial.suggest_int("buffer_size", 10000, 100000, step=10000)
    gamma = trial.suggest_float("gamma", 0.9, 0.99999)
    tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 10)
    
    # 환경 생성 (커리큘럼 파라미터는 고정값 사용)
    env = Environment(
        num_rays=8,
        initial_threshold=50,      # 고정값
        threshold_increment=20,    # 고정값
        window_size=10            # 고정값
    )
    
    # 모델 생성
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        verbose=0
    )
    
    try:
        # 모델 학습
        model.learn(total_timesteps=10000)
        
        # 모델 평가
        mean_reward = evaluate_model(model, env)
        print(f"\nTrial {trial.number} evaluation reward: {mean_reward:.2f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return float('-inf')
    finally:
        env.close()
    
    return mean_reward

def visualize_study(study, save_dir):
    """학습 과정 시각화 및 저장"""
    timestamp = int(time.time())
    
    # 최적화 히스토리
    fig_history = plot_optimization_history(study)
    fig_history.write_image(os.path.join(save_dir, f"optuna_history_{timestamp}.png"))
    
    # 파라미터 중요도
    fig_importance = plot_param_importances(study)
    fig_importance.write_image(os.path.join(save_dir, f"optuna_importance_{timestamp}.png"))
    
    # 파라미터 관계
    fig_parallel = plot_parallel_coordinate(study)
    fig_parallel.write_image(os.path.join(save_dir, f"optuna_parallel_{timestamp}.png"))
    
    # 파라미터 슬라이스
    fig_slice = plot_slice(study)
    fig_slice.write_image(os.path.join(save_dir, f"optuna_slice_{timestamp}.png"))

if __name__ == "__main__":
    save_dir = "A_SAC"
    os.makedirs(save_dir, exist_ok=True)
    
    study = optuna.create_study(direction="maximize")
    
    try:
        for i in range(30):
            study.optimize(objective, n_trials=1)
            
            print(f"\nTrial {i+1}/30 completed")
            print(f"Best value so far: {study.best_value:.2f}")
            print("Best params so far:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
            
            if (i + 1) % 5 == 0:  # 5번째 trial마다 시각화
                print("\nUpdating visualization plots...")
                visualize_study(study, save_dir)
    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    finally:
        print("\nSaving final results...")
        visualize_study(study, save_dir)
        
        df = study.trials_dataframe()
        csv_path = os.path.join(save_dir, "optuna_results_final.csv")
        df.to_csv(csv_path, index=False)
        
        best_params = {
            **study.best_trial.params,
            "best_value": study.best_value
        }
        json_path = os.path.join(save_dir, "best_params_final.json")
        with open(json_path, "w") as f:
            json.dump(best_params, f, indent=4)
        
        print(f"All results saved in '{save_dir}' directory!")
