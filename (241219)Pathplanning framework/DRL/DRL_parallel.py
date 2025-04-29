import numpy as np
from env import Environment
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import os
import multiprocessing as mp
from copy import deepcopy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 193)
        self.fc2 = nn.Linear(193, 341)
        self.fc3 = nn.Linear(341, 360)
        self.fc4 = nn.Linear(360, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 193)
        self.fc2 = nn.Linear(193, 341)
        self.fc3 = nn.Linear(341, 360)
        self.fc4 = nn.Linear(360, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, 
                 lr=0.000254704860583659,     # 최적화된 learning rate
                 gamma=0.9531584395834853,    # 최적화된 gamma
                 tau=0.046735309536096226):   # 최적화된 tau
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = []
        self.max_buffer_size = 100000

        self.noise_std = 0.24759172246362193
        self.noise_decay = 0.9928777479669256

        self.critic_losses = []
        self.actor_losses = []
        self.episode_rewards = []
        self.avg_rewards = []

        # 모델 저장을 위한 디렉토리 생성
        self.save_dir = 'models'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 최고 성능 기록
        self.best_reward = -float('inf')

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).numpy()[0]
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1, 1)
            
        self.noise_std *= self.noise_decay
        self.noise_std = max(0.05, self.noise_std)
        
        return action

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]

        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.FloatTensor(np.array([b[1] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor(np.array([float(b[4]) for b in batch])).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        
        return critic_loss.item(), actor_loss.item()

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def save_models(self, episode_reward, episode):
        """모델 저장 함수"""
        # 최고 성능 모델 저장
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'episode': episode,
                'reward': episode_reward,
            }, os.path.join(self.save_dir, 'best_model.pth'))
            print(f"\nNew best model saved! Reward: {episode_reward:.2f}")
        
        # 주기적으로 체크포인트 저장
        if episode % 100 == 0:  # 100 에피소드마다 저장
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'episode': episode,
                'reward': episode_reward,
            }, os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pth'))
    
    def save_training_progress(self):
        """학습 과정을 이미지로 저장하는 함수"""
        plt.figure(figsize=(15, 10))
        
        # 에피소드 리워드 플롯
        plt.subplot(3, 1, 1)
        plt.plot(self.episode_rewards, 'b-', label='Episode Reward')
        plt.plot(self.avg_rewards, 'r-', label='Average Reward')
        plt.grid(True)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Training Progress')
        
        # Critic 손실값 플롯
        plt.subplot(3, 1, 2)
        plt.plot(self.critic_losses, 'g-', label='Critic Loss')
        plt.grid(True)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        
        # Actor 손실값 플롯
        plt.subplot(3, 1, 3)
        plt.plot(self.actor_losses, 'm-', label='Actor Loss')
        plt.grid(True)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'))
        plt.close()


def train_agent(ray_length):
    """각각의 ray_length에 대한 학습을 수행하는 함수"""
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    # ray_length 별로 다른 저장 경로 설정
    save_dir = os.path.join('models', f'ray_{ray_length}')  # models/ray_3, models/ray_4, models/ray_5
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = Environment(max_ray_length=ray_length)
    state_dim = 4 + env.num_rays
    action_dim = 2

    agent = DDPGAgent(state_dim, action_dim)
    agent.save_dir = save_dir  # 저장 경로 업데이트
    
    episodes = 10000
    max_steps = 5000
    window_size = 10
    
    try:
        print(f"Starting training with ray_length = {ray_length}")
        for episode in range(episodes):
            observation = env.reset()
            if observation is None or 'robot_position' not in observation or 'rays' not in observation:
                raise ValueError(f"Invalid observation after reset: {observation}")

            state = np.concatenate([
                observation['robot_position'], 
                [observation['distance_to_goal']],
                [observation['angle_to_goal']],
                [ray[2] for ray in observation['rays']]
            ])
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < max_steps:
                # Pygame 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # 환경 렌더링 (필요한 경우에만)
                if episode % 100 == 0:  # 100 에피소드마다만 렌더링
                    screen.fill((255, 255, 255))
                    env.render_with_rays(screen)
                    pygame.display.flip()
                
                action = agent.act(state)
                action = np.clip(action, -1, 1)
                
                next_observation, reward, done = env.step(action)
                if next_observation is None or 'robot_position' not in next_observation or 'rays' not in next_observation:
                    raise ValueError(f"Invalid next observation after step: {next_observation}")

                next_state = np.concatenate([
                    next_observation['robot_position'], 
                    [next_observation['distance_to_goal']],
                    [next_observation['angle_to_goal']],
                    [ray[2] for ray in next_observation['rays']]
                ])
                
                agent.store_transition(state, action, reward, next_state, done)
                
                if len(agent.replay_buffer) > 64:
                    agent.train()
                
                state = next_state
                episode_reward += reward
                step_count += 1

                if episode % 100 == 0:  # 100 에피소드마다만 clock 제어
                    clock.tick(30)

            if step_count >= max_steps:
                print(f"Ray {ray_length} - Episode {episode + 1}: Maximum steps ({max_steps}) reached")
            
            agent.episode_rewards.append(episode_reward)
            
            if len(agent.episode_rewards) >= window_size:
                avg_reward = np.mean(agent.episode_rewards[-window_size:])
            else:
                avg_reward = episode_reward
            agent.avg_rewards.append(avg_reward)

            print(f"Ray {ray_length} - Episode {episode + 1}: Steps = {step_count}, "
                  f"Total Reward = {episode_reward:.2f}, Average Reward = {avg_reward:.2f}")
            
            # 모델 저장
            agent.save_models(episode_reward, episode)
            
            # 학습 진행 상황 주기적 저장
            if (episode + 1) % 100 == 0:
                agent.save_training_progress()
    
    except KeyboardInterrupt:
        print(f"\nTraining interrupted for ray_length = {ray_length}")
    except Exception as e:
        print(f"Error during training with ray_length = {ray_length}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent.save_training_progress()
        print(f"\nFinal training progress saved for ray_length = {ray_length}")
        pygame.quit()

if __name__ == "__main__":
    # 멀티프로세싱 설정
    ray_lengths = [3, 4, 5]  # 실험할 ray length 값들
    processes = []
    
    try:
        # 각 ray_length에 대해 별도의 프로세스 생성
        for ray_length in ray_lengths:
            p = mp.Process(target=train_agent, args=(ray_length,))
            p.start()
            processes.append(p)
        
        # 모든 프로세스가 완료될 때까지 대기
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        for p in processes:
            p.terminate()
    
    print("All training processes completed")