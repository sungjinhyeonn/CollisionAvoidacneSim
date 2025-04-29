import torch
import numpy as np
import matplotlib.pyplot as plt
from network import ActorCritic
from custom_nav_env import Navigation2DEnv
import time

class ModelEvaluator:
    def __init__(self, model_path):
        # 환경 설정
        self.env = Navigation2DEnv()
        self.state_dim = self.env.state_dim
        self.action_dim = 2
        
        # 모델 로드
        self.model = ActorCritic(self.state_dim, self.action_dim)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 시각화 설정
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        
        print(f"Loaded model from {model_path}")
        print(f"Best reward achieved during training: {checkpoint['reward']:.2f}")
        print(f"Timestep when best model was saved: {checkpoint['timestep']}")
    
    def evaluate_episode(self, render=True):
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        trajectory = [self.env.robot_pos.copy()]
        
        while not done and steps < 1000:
            # 행동 선택 (결정적)
            with torch.no_grad():
                action = self.model.get_action(state, deterministic=True)
            
            # 환경과 상호작용
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            steps += 1
            
            trajectory.append(self.env.robot_pos.copy())
            
            if render:
                self._render_episode(trajectory, steps, total_reward)
        
        return total_reward, steps, done
    
    def _render_episode(self, trajectory, steps, total_reward):
        plt.clf()
        
        # 궤적 그리기
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5, label='Robot Path')
        
        # 시작점과 현재 로봇 위치
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'bo', label='Start')
        plt.plot(self.env.robot_pos[0], self.env.robot_pos[1], 'ro', label='Robot')
        
        # 목표 위치
        plt.plot(self.env.goal_pos[0], self.env.goal_pos[1], 'g*', markersize=15, label='Goal')
        
        # 장애물
        for obs in self.env.obstacles:
            plt.plot(obs[0], obs[1], 'kx', markersize=10)
        
        # 그래프 설정
        plt.xlim(0, self.env.size)
        plt.ylim(0, self.env.size)
        plt.title(f'Step: {steps}, Reward: {total_reward:.2f}')
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.01)
    
    def run_evaluation(self, num_episodes=100):
        """여러 에피소드에 대해 평가 수행"""
        rewards = []
        success_count = 0
        collision_count = 0
        steps_list = []
        
        print("\nStarting evaluation...")
        for episode in range(num_episodes):
            reward, steps, done = self.evaluate_episode(render=(episode % 10 == 0))
            rewards.append(reward)
            steps_list.append(steps)
            
            # 성공/충돌 판정
            if reward >= 100:
                success_count += 1
            elif reward <= -100:
                collision_count += 1
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
                print(f"- Reward: {reward:.2f}")
                print(f"- Steps: {steps}")
                print(f"- Success: {'Yes' if reward >= 100 else 'No'}")
                time.sleep(1)  # 시각화를 위한 대기
        
        # 최종 성능 통계
        print("\n=== Evaluation Results ===")
        print(f"Number of episodes: {num_episodes}")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Average steps: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
        print(f"Success rate: {success_count/num_episodes*100:.1f}%")
        print(f"Collision rate: {collision_count/num_episodes*100:.1f}%")
        
        # 결과 히스토그램 표시
        plt.figure(figsize=(10, 5))
        plt.hist(rewards, bins=20)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Count')
        plt.show()

if __name__ == "__main__":
    evaluator = ModelEvaluator('best_navigation_model.pth')
    evaluator.run_evaluation(num_episodes=100) 