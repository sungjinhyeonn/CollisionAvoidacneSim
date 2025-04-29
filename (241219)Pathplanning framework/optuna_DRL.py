import optuna
import numpy as np
from DRL.env import Environment
from DRL.DRL import DDPGAgent
import torch
import os

def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    tau = trial.suggest_float('tau', 0.001, 0.1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    noise_std = trial.suggest_float('noise_std', 0.1, 0.5)
    noise_decay = trial.suggest_float('noise_decay', 0.99, 0.9999)
    
    # 환경 및 에이전트 초기화
    env = Environment(num_rays=8)
    state_dim = 4 + env.num_rays  # robot state + ray distances
    action_dim = 2  # linear and angular velocity
    
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        tau=tau
    )
    agent.noise_std = noise_std
    agent.noise_decay = noise_decay
    
    # 평가 에피소드 실행
    n_episodes = 50  # 최적화 시 사용할 에피소드 수
    total_rewards = []
    
    for episode in range(n_episodes):
        observation = env.reset()
        if observation is None:
            continue
            
        state = np.concatenate([
            observation['robot_position'],
            [observation['distance_to_goal']],
            [observation['angle_to_goal']],
            [ray[2] for ray in observation['rays']]
        ])
        
        episode_reward = 0
        done = False
        step_count = 0
        max_steps = 1000
        
        while not done and step_count < max_steps:
            action = agent.act(state)
            next_observation, reward, done = env.step(action)
            
            if next_observation is None:
                break
                
            next_state = np.concatenate([
                next_observation['robot_position'],
                [next_observation['distance_to_goal']],
                [next_observation['angle_to_goal']],
                [ray[2] for ray in next_observation['rays']]
            ])
            
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > batch_size:
                agent.train(batch_size=batch_size)
            
            state = next_state
            episode_reward += reward
            step_count += 1
        
        total_rewards.append(episode_reward)
        
        # 중간 결과 보고
        trial.report(np.mean(total_rewards), episode)
        
        # 성능이 너무 안 좋으면 조기 종료
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(total_rewards)

def optimize_hyperparameters():
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
        study_name='ddpg_optimization'
    )
    
    study.optimize(objective, n_trials=100, timeout=None)
    
    # 최적의 하이퍼파라미터 출력
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 결과 저장
    if not os.path.exists('optuna_results'):
        os.makedirs('optuna_results')
    
    # 최적화 과정 시각화
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("optuna_results/optimization_history.png")
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("optuna_results/param_importances.png")
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image("optuna_results/parallel_coordinate.png")
    except:
        print("Could not generate visualization plots")
    
    return study.best_params

if __name__ == "__main__":
    best_params = optimize_hyperparameters() 