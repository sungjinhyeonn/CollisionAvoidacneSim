import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from shapely.geometry import Point, Polygon, LineString
import os
import logging
import datetime

class NeuralNet(nn.Module):
    def __init__(self, input_dim=6):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class Neural_RRT:
    def __init__(self):
        # 로깅 설정
        self.setup_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuralNet(input_dim=6).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_data = []
        self.training_labels = []
        self.total_iterations = 0
        self.successful_expansions = 0
        self.training_count = 0
        self.save_interval = 10  # 100번의 성공적인 경로 생성마다 학습 및 저장
        self.is_trained = False  # 학습 여부 플래그 추가
        
        # 모델 저장 경로
        self.model_path = 'saved_models/neural_rrt_model_20_20241119_145600.pth'
        self.load_model()
    
    def setup_logger(self):
        """로깅 설정"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{log_dir}/neural_rrt_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def save_model(self):
        """모델 저장"""
        os.makedirs('saved_models', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'saved_models/neural_rrt_model_{self.training_count}_{timestamp}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_data': self.training_data,
            'training_labels': self.training_labels,
            'training_count': self.training_count,
            'total_iterations': self.total_iterations,
            'successful_expansions': self.successful_expansions
        }, save_path)
        
        self.logger.info(f'Model saved to {save_path}')
        
    def load_model(self):
        """저장된 모델 불러오기"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_data = checkpoint['training_data']
                self.training_labels = checkpoint['training_labels']
                self.training_count = checkpoint.get('training_count', 0)
                self.total_iterations = checkpoint.get('total_iterations', 0)
                self.successful_expansions = checkpoint.get('successful_expansions', 0)
                self.is_trained = True  # 모델 로드 성공
                self.logger.info("Loaded saved model - Training will be skipped")
            else:
                self.logger.info("No saved model found - Will start training from scratch")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.info("Will start training from scratch")

    def predict_sample(self, start, goal, current):
        self.model.eval()
        with torch.no_grad():
            input_data = torch.FloatTensor(
                np.concatenate([start, goal, current])
            ).to(self.device)
            prediction = self.model(input_data).cpu().numpy()
        self.model.train()
        return prediction

    def collect_data(self, start, goal, successful_point, current_point):
        input_data = np.concatenate([start, goal, current_point])
        self.training_data.append(input_data)
        self.training_labels.append(successful_point)
        
        max_memory = 10000
        if len(self.training_data) > max_memory:
            self.training_data = self.training_data[-max_memory:]
            self.training_labels = self.training_labels[-max_memory:]
        
    def train(self, epochs=10, batch_size=32):
        """학습 및 모델 저장"""
        if len(self.training_data) > batch_size:
            self.logger.info(f"Training with {len(self.training_data)} samples")
            X = torch.FloatTensor(self.training_data).to(self.device)
            y = torch.FloatTensor(self.training_labels).to(self.device)
            
            total_loss = 0
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                self.logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')
            
            avg_loss = total_loss / epochs
            self.logger.info(f'Training completed. Average Loss: {avg_loss:.6f}')
            
            # 모델 저장
            self.save_model()
            self.logger.info(f'Model saved after {self.training_count} successful paths')

    @staticmethod
    def simplify_path(path, terrain_polygons):
        """LOS 기반 경로 단순화"""
        if not path or len(path) < 3:
            return path
            
        def is_visible(start, end, polygons):
            """두 점 사이의 직선 경로가 가능한지 확인"""
            line = LineString([start, end])
            for polygon in polygons:
                if isinstance(polygon, list):
                    poly = Polygon(polygon)
                else:
                    poly = polygon
                if poly.intersects(line):
                    return False
            return True

        simplified = [path[0]]  # 시작점 추가
        current_idx = 0

        while current_idx < len(path):
            # 현재 점에서 가장 멀리 보이는 점 찾기
            farthest_visible_idx = current_idx
            for i in range(len(path)-1, current_idx, -1):
                if is_visible(path[current_idx], path[i], terrain_polygons):
                    farthest_visible_idx = i
                    break
            
            if farthest_visible_idx == current_idx:
                current_idx += 1
            else:
                simplified.append(path[farthest_visible_idx])
                current_idx = farthest_visible_idx

        return simplified

    def plan_path(self, start, goal, terrain_polygons, max_iterations=1000, step_size=1):
        path = None
        nodes = [Node(start)]
        
        def collision_free(from_point, to_point, polygons):
            """두 점 사이의 경로가 장애물과 충돌하는지 확인"""
            line = LineString([from_point, to_point])
            for polygon in polygons:
                if isinstance(polygon, list):  # 폴리곤이 좌표 리스트로 주어진 경우
                    poly = Polygon(polygon)
                else:  # 이미 Polygon 객체인 경우
                    poly = polygon
                    
                if poly.intersects(line):
                    return False
            return True
        
        for i in range(max_iterations):
            # 학습된 모델이 있으면 더 높은 확률로 신경망 사용
            use_neural = (np.random.random() < 0.9) if self.is_trained else (
                np.random.random() < 0.7 and len(self.training_data) > 100
            )
            
            if use_neural:
                current = nodes[-1].point
                sample = self.predict_sample(
                    np.array(start), 
                    np.array(goal), 
                    np.array(current)
                )
            else:
                x = np.random.uniform(min(start[0], goal[0])-10, max(start[0], goal[0])+10)
                y = np.random.uniform(min(start[1], goal[1])-10, max(start[1], goal[1])+10)
                sample = (x, y)

            nearest = self.nearest_node(nodes, sample)
            new_point = self.steer(nearest, sample, step_size)
            
            if collision_free(nearest.point, new_point, terrain_polygons):
                new_node = Node(new_point, nearest)
                nodes.append(new_node)
                
                # 학습된 모델이 없을 때만 데이터 수집
                if not self.is_trained:
                    self.collect_data(
                        np.array(start),
                        np.array(goal),
                        np.array(new_point),
                        np.array(nearest.point)
                    )
                
                if self.distance(new_point, goal) < step_size:
                    raw_path = self.reconstruct_path(new_node)
                    # LOS 기반 경로 단순화 적용
                    simplified_path = self.simplify_path(raw_path, terrain_polygons)
                    self.training_count += 1
                    
                    # 학습된 모델이 없을 때만 추가 학습
                    if not self.is_trained and self.training_count % self.save_interval == 0:
                        self.logger.info(f"Starting training after {self.training_count} successful paths")
                        self.train(epochs=5, batch_size=32)
                        
                    self.logger.info(f"Path simplified from {len(raw_path)} to {len(simplified_path)} points")
                    return simplified_path
        
        return None

    @staticmethod
    def distance(p1, p2):
        """두 점 사이의 유클리드 거리 계산"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def nearest_node(nodes, point):
        """주어진 점에서 가장 가까운 노드 찾기"""
        return min(nodes, key=lambda n: Neural_RRT.distance(n.point, point))

    @staticmethod
    def steer(from_node, to_point, step_size):
        """현재 노드에서 목표점을 향해 step_size만큼 이동"""
        if Neural_RRT.distance(from_node.point, to_point) < step_size:
            return to_point
        direction = np.array(to_point) - np.array(from_node.point)
        direction = direction / np.linalg.norm(direction)
        new_point = np.array(from_node.point) + step_size * direction
        return tuple(new_point)

    @staticmethod
    def reconstruct_path(node):
        """노드로부터 시작점까지의 경로 재구성"""
        path = []
        current = node
        while current is not None:
            path.append(current.point)
            current = current.parent
        return path[::-1]

class Node:
    """RRT 트리의 노드"""
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent