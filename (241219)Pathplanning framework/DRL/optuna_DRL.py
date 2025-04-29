import optuna
import numpy as np
from env import Environment
from DRL import DDPGAgent
import torch
import torch.nn as nn
import os

def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    tau = trial.suggest_float('tau', 0.001, 0.1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)
    noise_std = trial.suggest_float('noise_std', 0.1, 0.5)
    noise_decay = trial.suggest_float('noise_decay', 0.99, 0.9999)
    
    # 네트워크 구조를 위한 하이퍼파라미터
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_sizes = []
    for i in range(n_layers):
        hidden_sizes.append(trial.suggest_int(f'hidden_size_layer_{i}', 64, 512))
    
    # 에피소드 관련 파라미터
    n_episodes = trial.suggest_int('n_episodes', 5000, 15000)
    max_steps_per_episode = trial.suggest_int('max_steps_per_episode', 3000, 7000)
    
    # Actor 네트워크 정의
    class TrialActor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(TrialActor, self).__init__()
            layers = []
            prev_dim = state_dim
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_dim, hidden_size))
                layers.append(nn.ReLU())
                prev_dim = hidden_size
            
            layers.append(nn.Linear(prev_dim, action_dim))
            layers.append(nn.Tanh())
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, state):
            return self.model(state)
    
    # Critic 네트워크 정의
    class TrialCritic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(TrialCritic, self).__init__()
            layers = []
            prev_dim = state_dim + action_dim
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_dim, hidden_size))
                layers.append(nn.ReLU())
                prev_dim = hidden_size
            
            layers.append(nn.Linear(prev_dim, 1))
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            return self.model(x)
    
    # 환경 및 에이전트 초기화
    env = Environment(num_rays=8)
    state_dim = 4 + env.num_rays
    action_dim = 2
    
    # 커스텀 DDPGAgent 클래스 정의
    class TrialDDPGAgent(DDPGAgent):
        def __init__(self, state_dim, action_dim, lr, gamma, tau):
            super().__init__(state_dim, action_dim, lr, gamma, tau)
            self.actor = TrialActor(state_dim, action_dim)
            self.actor_target = TrialActor(state_dim, action_dim)
            self.critic = TrialCritic(state_dim, action_dim)
            self.critic_target = TrialCritic(state_dim, action_dim)
            
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
    
    agent = TrialDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        tau=tau
    )
    agent.noise_std = noise_std
    agent.noise_decay = noise_decay
    
    # 평가 에피소드 실행
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
        
        while not done and step_count < max_steps_per_episode:
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
        
        # 조기 종료 조건
        if len(total_rewards) > 100:
            recent_avg = np.mean(total_rewards[-100:])
            if recent_avg > 200:  # 임계값 설정
                break
        
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
    
    study.optimize(objective, n_trials=50, timeout=None)
    
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