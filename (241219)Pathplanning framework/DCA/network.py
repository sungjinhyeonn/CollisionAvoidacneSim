import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        print(f"Creating ActorCritic with state_dim: {state_dim}, action_dim: {action_dim}")
        
        # 상태 및 행동 차원 저장
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor 네트워크
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 평균과 표준편차를 위한 출력 레이어
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
        # Critic 네트워크
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 행동 로그 표준편차의 범위 제한
        self.log_std_min = -20
        self.log_std_max = 2
        
        print(f"Network architecture created with state_dim: {state_dim}")
    
    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Actor
        actor_features = self.actor(state)
        mean = self.mean_layer(actor_features)
        log_std = self.log_std_layer(actor_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # Critic
        value = self.critic(state)
        
        return mean, std, value
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            mean, std, _ = self.forward(state)
            
            if deterministic:
                # 목표 근처에서는 더 작은 행동 생성
                action = mean
                state_tensor = torch.FloatTensor(state)
                goal_distance = torch.norm(state_tensor[:, :2])  # 로봇-목표 상대위치
                if goal_distance < 2.0:  # 임계값 조정 가능
                    action *= 0.5  # 스케일링 팩터 조정 가능
                return action.numpy()[0]
            
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action.numpy()[0]
    
    def evaluate_actions(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
            
        mean, std, value = self.forward(state)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value.squeeze()
