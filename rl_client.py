# 强化学习客户端
import socket
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 定义Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(state_size, action_size).to(self.device)
        self.target_dqn = DQN(state_size, action_size).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.memory = ReplayBuffer(100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_frequency = 1000
        self.train_frequency = 4
        self.steps = 0
        self.last_action_time = time.time()
        print("Agent初始化完成")

    def act(self, state):
        if random.random() < self.epsilon:
            if random.random() < 0.3:  # 30%的概率选择发射动作
                return 2  # 发射动作的索引
            else:
                return random.randint(0, 1)  # 随机选择左转或右转
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.dqn(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.steps += 1
        if self.steps % self.train_frequency != 0:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.steps % self.update_target_frequency == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        print("目标网络已更新")

# 连接到服务器
def connect_to_server():
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('localhost', 5555))
            print("成功连接到服务器")
            return client_socket
        except Exception as e:
            print(f"连接服务器失败: {e}")
            print("5秒后重试...")
            time.sleep(5)

# 主函数
def main():
    client_socket = connect_to_server()
    state_size = 7  # 假设状态包括发射器角度、颜色、球的数量、分数、游戏是否结束、难度和路径长度
    action_size = 3  # 左转、右转、发射
    agent = DQNAgent(state_size, action_size)

    start_time = time.time()
    last_action_time = time.time()
    while True:
        print("开始得分更新")
        state = get_game_state(client_socket)
        total_reward = 0
        done = False

        while not done:
            current_time = time.time()
            if current_time - last_action_time >= 0.2:
                action = agent.act(state)
                if action == 0:
                    success, response = send_action(client_socket, 'rotate', -1)
                elif action == 1:
                    success, response = send_action(client_socket, 'rotate', 1)
                else:
                    success, response = send_action(client_socket, 'shoot')
                
                print(f"操作: {'左转' if action == 0 else '右转' if action == 1 else '发射'}")
                print(f"操作成功: {success}")
                print(f"服务器响应: {response}")
                
                last_action_time = current_time

                next_state = get_game_state(client_socket)
                reward = calculate_reward(state, next_state, agent)
                done = next_state[4]

                agent.memory.push(state, action, reward, next_state, done)
                agent.train()

                state = next_state
                total_reward += reward

            if time.time() - start_time >= 5:
                break

        print(f"得分更新结束，总奖励: {total_reward}，当前epsilon: {agent.epsilon:.4f}")

        # 每100次得分更新保存模型
        if int((time.time() - start_time) / 1) % 100 == 0:
            torch.save(agent.dqn.state_dict(), f'dqn_model_update_{int((time.time() - start_time) / 5)}.pth')

    client_socket.close()
    print("游戏训练结束")

# 获取游戏状态
def get_game_state(client_socket):
    client_socket.send(json.dumps({'action': 'get_state'}).encode('utf-8'))
    data = b''
    while True:
        chunk = client_socket.recv(4096)
        data += chunk
        try:
            game_state = json.loads(data.decode('utf-8'))
            break
        except json.JSONDecodeError:
            continue
    state = [
        game_state['shooter_angle'] / 360.0,  # 归一化角度
        COLORS.index(tuple(game_state['shooter_color'])) / len(COLORS),  # 颜色索引归一化
        len(game_state['balls']) / 100,  # 假设最多100个球
        game_state['score'] / 1000,  # 假设最高分1000
        1 if game_state['game_over'] else 0,
        game_state['difficulty'] / 10,  # 假设最高难度10
        len(game_state['path']) / 1000  # 假设最长路径1000
    ]
    return state

# 发送操作到服务器
def send_action(client_socket, action, direction=None):
    try:
        if direction is not None:
            client_socket.send(json.dumps({'action': action, 'direction': direction}).encode('utf-8'))
        else:
            client_socket.send(json.dumps({'action': action}).encode('utf-8'))
        
        # 等待服务器响应
        response = client_socket.recv(4096).decode('utf-8')
        return True, response
    except Exception as e:
        print(f"发送操作失败: {e}")
        return False, str(e)

# 计算奖励
def calculate_reward(state, next_state, agent):
    score_diff = (next_state[3] - state[3]) * 1000  # 分数差
    ball_diff = (state[2] - next_state[2]) * 100  # 球数量减少
    game_over_penalty = -100 if next_state[4] else 0  # 游戏结束惩罚
    
    return score_diff + ball_diff + game_over_penalty

# 定义颜色列表（与服务器端一致）
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

if __name__ == "__main__":
    main()
