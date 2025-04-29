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
from sac_env import Environment
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

def create_env():
    env = Environment(
        grid_size=(20, 20),
        num_dynamic_obstacles=5,  # 시작 장애물 수 5개
        num_rays=8,
        max_ray_length=3,
        consecutive_successes_required=20  # 연속 성공 20번시 난이도 상승
    )
    return env

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
    
    # 환경 생성
    env = Environment(
        grid_size=(20, 20),
        num_dynamic_obstacles=5,  # 시작 장애물 수 5개
        num_rays=8,
        max_ray_length=3,
        consecutive_successes_required=20
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
        
        # 현재까지의 최고 성능이면 모델 저장
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        if trial.number == 0 or mean_reward > trial.study.best_value:
            model_dir = "A_SAC/models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"best_model_trial_{trial.number}_reward_{mean_reward:.2f}")
            model.save(model_path)
            print(f"New best model saved to {model_path}!")
            
            # 하이퍼파라미터도 저장
            params = {
                "trial_number": trial.number,
                "reward": mean_reward,
                "parameters": trial.params
            }
            params_path = os.path.join("A_SAC", f"best_params_trial_{trial.number}.json")
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)
            
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
            
            # if (i + 1) % 5 == 0:  # 5번째 trial마다 시각화
            #     print("\nUpdating visualization plots...")
            #     visualize_study(study, save_dir)
    
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    finally:
        print("\nSaving final results...")
        visualize_study(study, save_dir)
        
        # 최종 결과 저장
        df = study.trials_dataframe()
        df.to_csv(os.path.join(save_dir, "optuna_results_final.csv"), index=False)
        
        best_params = {
            "trial_number": study.best_trial.number,
            "reward": study.best_value,
            "parameters": study.best_trial.params
        }
        with open(os.path.join(save_dir, "best_params_final.json"), "w") as f:
            json.dump(best_params, f, indent=4)
        
        print(f"All results saved in '{save_dir}' directory!")
