import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import ActorCritic
from custom_nav_env import Navigation2DEnv
import matplotlib.pyplot as plt
import time

class PPOTrainer:
    def __init__(self):
        # 하이퍼파라미터
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.epochs = 10
        self.max_steps = 1000
        self.batch_size = 64
        
        # 환경 및 모델 설정
        self.env = Navigation2DEnv()
        
        # 상태 차원 계산 (환경에서 직접 가져옴)
        self.state_dim = self.env.state_dim
        self.action_dim = 2
        
        print(f"PPOTrainer initialization:")
        print(f"- State dimension: {self.state_dim}")
        print(f"- Action dimension: {self.action_dim}")
        
        # 모델 초기화
        self.model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.fig = plt.figure(figsize=(8, 8))
        plt.ion()  # 대화형 모드 활성화
        
        # 커리큘럼 학습 파라미터
        self.curriculum_stages = {
            0: {"num_obstacles": 2, "max_velocity": 0.2, "max_goal_distance": 20},
            100000: {"num_obstacles": 3, "max_velocity": 0.3, "max_goal_distance": 25},
            300000: {"num_obstacles": 4, "max_velocity": 0.4, "max_goal_distance": 30},
            500000: {"num_obstacles": 5, "max_velocity": 0.5, "max_goal_distance": 40}
        }
        self.current_stage = 0
    
    def get_current_difficulty(self, timestep):
        """현재 타임스텝에 맞는 난이도 설정을 반환"""
        current_stage = 0
        for stage_timestep in sorted(self.curriculum_stages.keys()):
            if timestep >= stage_timestep:
                current_stage = stage_timestep
            else:
                break
        return self.curriculum_stages[current_stage]
    
    def update_env_difficulty(self, timestep):
        """환경의 난이도를 현재 학습 단계에 맞게 업데이트"""
        difficulty = self.get_current_difficulty(timestep)
        self.env.update_difficulty(
            difficulty["num_obstacles"],
            difficulty["max_velocity"],
            difficulty["max_goal_distance"]
        )
    
    def collect_rollouts(self, num_steps):
        states, actions, rewards = [], [], []
        values, log_probs, dones = [], [], []
        
        state = self.env.reset()
        done = False
        episode_reward = 0
        
        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 행동 선택 및 가치 계산
            with torch.no_grad():
                action_mean, action_std, value = self.model(state_tensor)
                action = self.model.get_action(state)
                log_prob, _, _ = self.model.evaluate_actions(state_tensor, 
                                                           torch.FloatTensor(action).unsqueeze(0))
            
            # 환경과 상호작용
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            
            # 데이터 저장
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            state = next_state
            
            if done:
                state = self.env.reset()
                done = False
        
        return (torch.FloatTensor(states), 
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(values),
                torch.FloatTensor(log_probs),
                torch.FloatTensor(dones)), episode_reward
    
    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            
        returns = advantages + values
        return advantages, returns
    
    def visualize_episode(self, render_steps=200):
        """에피소드 실행 및 시각화"""
        state = self.env.reset()
        trajectory = [self.env.robot_pos.copy()]
        
        plt.clf()
        for step in range(render_steps):
            with torch.no_grad():
                action = self.model.get_action(state, deterministic=True)
            
            state, reward, done, _ = self.env.step(action)
            trajectory.append(self.env.robot_pos.copy())
            
            # 시각화
            plt.clf()
            plt.xlim(0, self.env.size)
            plt.ylim(0, self.env.size)
            
            # 궤적 그리기
            trajectory_array = np.array(trajectory)
            plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', alpha=0.5, label='Robot Trajectory')
            
            # 현재 로봇 위치
            plt.plot(self.env.robot_pos[0], self.env.robot_pos[1], 'bo', markersize=10, label='Robot')
            
            # 목표 위치
            plt.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'g*', markersize=15, label='Goal')
            
            # 장애물 (이제 동적으로 움직임)
            for obs in self.env.obstacles:
                plt.plot(obs[0], obs[1], 'rx', markersize=10)
            
            plt.legend()
            plt.title(f'Step: {step}, Reward: {reward:.2f}')
            plt.draw()
            plt.pause(0.01)
            
            if done:
                break
    
    def train(self, total_timesteps):
        timesteps_per_batch = 2048
        total_episodes = 0
        best_reward = -np.inf
        visualization_interval = 100
        
        # 로깅을 위한 변수들
        rewards_history = []
        episode_length_history = []
        collision_history = []
        success_history = []
        
        print("\n=== Training Start ===")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Batch size: {timesteps_per_batch}")
        
        for timestep in range(0, total_timesteps, timesteps_per_batch):
            # 난이도 업데이트
            self.update_env_difficulty(timestep)
            current_difficulty = self.get_current_difficulty(timestep)
            
            # 데이터 수집
            rollouts, episode_reward = self.collect_rollouts(timesteps_per_batch)
            states, actions, rewards, values, old_log_probs, dones = rollouts
            
            # GAE 계산
            advantages, returns = self.compute_gae(rewards, values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 성능 메트릭 계산
            num_collisions = sum([1 for r in rewards if r <= -100])
            num_successes = sum([1 for r in rewards if r >= 100])
            episode_length = len(rewards)
            
            # 히스토리 업데이트
            rewards_history.append(episode_reward)
            episode_length_history.append(episode_length)
            collision_history.append(num_collisions)
            success_history.append(num_successes)
            
            # 최근 100 에피소드의 평균 성능
            avg_reward = np.mean(rewards_history[-100:])
            avg_length = np.mean(episode_length_history[-100:])
            avg_collisions = np.mean(collision_history[-100:])
            avg_success_rate = np.mean(success_history[-100:]) if success_history else 0
            
            # 터미널 출력
            print("\n" + "="*50)
            print(f"Timestep: {timestep}/{total_timesteps}")
            print(f"Current Stage - Obstacles: {current_difficulty['num_obstacles']}, "
                  f"Max velocity: {current_difficulty['max_velocity']:.2f}, "
                  f"Max goal distance: {current_difficulty['max_goal_distance']}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Average Reward (100 eps): {avg_reward:.2f}")
            print(f"Average Episode Length: {avg_length:.1f}")
            print(f"Collisions in this batch: {num_collisions}")
            print(f"Successes in this batch: {num_successes}")
            print(f"Success Rate (100 eps): {avg_success_rate*100:.1f}%")
            
            # PPO 업데이트
            for _ in range(self.epochs):
                for idx in range(0, len(states), self.batch_size):
                    batch_states = states[idx:idx + self.batch_size]
                    batch_actions = actions[idx:idx + self.batch_size]
                    batch_advantages = advantages[idx:idx + self.batch_size]
                    batch_returns = returns[idx:idx + self.batch_size]
                    batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                    
                    # 현재 정책으로 행동 평가
                    new_log_probs, entropy, value_pred = self.model.evaluate_actions(
                        batch_states, batch_actions)
                    
                    # 비율 계산
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # PPO 손실 계산
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Critic 손실 계산
                    value_loss = 0.5 * (batch_returns - value_pred).pow(2).mean()
                    
                    # 전체 손실
                    loss = actor_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                    
                    # 역전파 및 최적화
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
            
            # 모델 저장 (성능이 향상된 경우)
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'reward': best_reward,
                    'timestep': timestep
                }, 'best_navigation_model.pth')
                print(f"New best model saved! Reward: {best_reward:.2f}")
            
            # 주기적으로 에이전트 시각화
            if timestep % visualization_interval == 0:
                print("\nVisualizing current policy...")
                self.visualize_episode()
                time.sleep(0.5)

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(total_timesteps=10000000)