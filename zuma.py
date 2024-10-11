import pygame
import random
import math
import socket
import json
import threading
import time

# 初始化Pygame
pygame.init()

# 设置窗口大小
WIDTH = 1024
HEIGHT = 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eddie's Zuma Game Server")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# 定义球的类
class Ball:
    def __init__(self, position, color):
        self.position = position
        self.color = color
        self.radius = 20
        self.speed = 1
        self.path_index = 0
        self.velocity = [0, 0]
        self.acceleration = [0, 0]

    def move(self, path, balls):
        if self.path_index < len(path) - 1:
            target = path[self.path_index + 1]
            direction = (target[0] - self.position[0], target[1] - self.position[1])
            distance = math.sqrt(direction[0]**2 + direction[1]**2)
            
            # 检查前方是否有球
            next_ball = next((ball for ball in balls if balls.index(ball) == balls.index(self) + 1), None)
            if next_ball:
                distance_to_next_ball = math.sqrt((next_ball.position[0] - self.position[0])**2 + 
                                                  (next_ball.position[1] - self.position[1])**2)
                if distance_to_next_ball <= 2 * self.radius:
                    return  # 如果太近，就不移动

            if distance > self.speed:
                self.acceleration = [direction[0] / distance * 0.1, direction[1] / distance * 0.1]
                self.velocity[0] += self.acceleration[0]
                self.velocity[1] += self.acceleration[1]
                self.velocity[0] *= 0.95  # 添加摩擦力
                self.velocity[1] *= 0.95
                new_position = (
                    self.position[0] + self.velocity[0],
                    self.position[1] + self.velocity[1]
                )
                
                # 检查新位置是否会导致重叠
                if not next_ball or math.sqrt((new_position[0] - next_ball.position[0])**2 + 
                                              (new_position[1] - next_ball.position[1])**2) > 2 * self.radius:
                    self.position = new_position
            else:
                self.position = target
                self.path_index += 1
                self.velocity = [0, 0]

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)

# 定义发射器类
class Shooter:
    def __init__(self):
        self.position = (WIDTH // 2, HEIGHT // 2)
        self.angle = 90
        self.color = random.choice(COLORS)
        self.reload_time = 0

    def draw(self):
        pygame.draw.circle(screen, self.color, self.position, 30)
        end_x = self.position[0] + 50 * math.cos(math.radians(self.angle))
        end_y = self.position[1] - 50 * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, BLACK, self.position, (end_x, end_y), 5)

    def rotate(self, direction):
        self.angle += direction * 5
        # 移除角度限制
        self.angle %= 360  # 保持角度在0-359范围内

    def shoot(self, path):
        if self.reload_time <= 0:
            start_pos = (
                self.position[0] + 50 * math.cos(math.radians(self.angle)),
                self.position[1] - 50 * math.sin(math.radians(self.angle))
            )
            self.reload_time = 30  # 设置重新装填时间
            return Ball(start_pos, self.color)
        return None

    def update(self):
        if self.reload_time > 0:
            self.reload_time -= 1

# 游戏状态类
class GameState:
    def __init__(self):
        self.shooter = Shooter()
        self.balls = []
        self.path = self.generate_spiral_path()
        self.score = 0
        self.spawn_timer = 0
        self.game_over = False
        self.ball_spacing = 45  # 球之间的间距，稍微减小以适应新的逻辑
        self.difficulty = 1

    def generate_spiral_path(self):
        path = []
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        radius = min(WIDTH, HEIGHT) // 2 - 50  # 从外圈开始
        angle = 0
        while radius > 10:
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            path.append((x, y))
            angle += 0.1
            radius -= 0.5
        return path

    def update(self):
        if self.game_over:
            return

        self.shooter.update()

        # 移动所有球
        for i, ball in enumerate(self.balls):
            ball.move(self.path, self.balls)

        # 检查是否有球到达终点
        if self.balls and self.balls[-1].path_index == len(self.path) - 1:
            self.game_over = True
            return

        # 检查是否有三个或更多相同颜色的球相邻
        i = 0
        while i < len(self.balls) - 2:
            if self.balls[i].color == self.balls[i+1].color == self.balls[i+2].color:
                # 移除匹配的球
                del self.balls[i:i+3]
                self.score += 30 * self.difficulty
                # 碰撞效果：前面的球后退
                for j in range(i):
                    self.balls[j].path_index = max(0, self.balls[j].path_index - 3)
                    self.balls[j].position = self.path[self.balls[j].path_index]
                    self.balls[j].velocity = [0, 0]  # 重置速度
            else:
                i += 1

        # 定期生成新球
        self.spawn_timer += 1
        if self.spawn_timer >= max(60 - self.difficulty * 5, 20):  # 根据难度调整生成速度
            self.spawn_timer = 0
            if not self.balls or self.can_spawn_new_ball():
                new_ball = Ball(self.path[0], random.choice(COLORS))
                new_ball.path_index = 0
                self.balls.insert(0, new_ball)  # 在开头插入新球

        # 增加难度
        self.difficulty += 0.001

    def can_spawn_new_ball(self):
        if not self.balls:
            return True
        first_ball = self.balls[0]
        return first_ball.path_index * self.ball_spacing > len(self.path) // 10

    def draw(self):
        screen.fill(WHITE)
        # 绘制路径
        pygame.draw.lines(screen, (200, 200, 200), False, self.path, 2)
        self.shooter.draw()
        for ball in self.balls:
            ball.draw()
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {int(self.score)}", True, BLACK)
        screen.blit(score_text, (10, 10))
        difficulty_text = font.render(f"Difficulty: {self.difficulty:.1f}", True, BLACK)
        screen.blit(difficulty_text, (10, 50))
        if self.game_over:
            game_over_text = font.render("Game Over", True, (255, 0, 0))
            screen.blit(game_over_text, (WIDTH//2 - 70, HEIGHT//2))

# 处理客户端请求的函数
def handle_client(client_socket, game_state):
    try:
        client_socket.settimeout(5)  # 设置5秒超时
        request_queue = []
        while True:
            try:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                command = json.loads(data)
                request_queue.append(command)

                while request_queue:
                    current_command = request_queue[0]
                    if current_command['action'] == 'rotate':
                        game_state.shooter.rotate(current_command['direction'])
                        request_queue.pop(0)
                    elif current_command['action'] == 'shoot':
                        new_ball = game_state.shooter.shoot(game_state.path)
                        if new_ball:
                            # 计算新球的射线
                            angle_rad = math.radians(game_state.shooter.angle)
                            direction = (math.cos(angle_rad), -math.sin(angle_rad))
                            
                            # 找到射线与现有球相交的点
                            insert_index = None
                            for i, ball in enumerate(game_state.balls):
                                # 计算射线与球的交点
                                dx = ball.position[0] - new_ball.position[0]
                                dy = ball.position[1] - new_ball.position[1]
                                a = direction[0]**2 + direction[1]**2
                                b = 2 * (dx * direction[0] + dy * direction[1])
                                c = dx**2 + dy**2 - (ball.radius * 1.5)**2  # 增加判定范围
                                discriminant = b**2 - 4*a*c
                                
                                if discriminant >= 0:
                                    # 射线与球相交
                                    insert_index = i
                                    break
                            
                            if insert_index is not None:
                                # 将新球插入到相交点的位置
                                game_state.balls.insert(insert_index, new_ball)
                                # 设置新球的路径索引
                                if insert_index > 0:
                                    prev_ball = game_state.balls[insert_index - 1]
                                    new_ball.path_index = prev_ball.path_index + 1
                                else:
                                    new_ball.path_index = 0
                                new_ball.position = game_state.path[new_ball.path_index]
                            else:
                                # 如果没有相交，将新球添加到队列末尾
                                game_state.balls.append(new_ball)
                                new_ball.path_index = 0
                                new_ball.position = game_state.path[0]
                            
                            game_state.shooter.color = random.choice(COLORS)
                        request_queue.pop(0)
                    elif current_command['action'] == 'get_state':
                        state = {
                            'shooter_angle': game_state.shooter.angle,
                            'shooter_color': game_state.shooter.color,
                            'balls': [(ball.position, ball.color) for ball in game_state.balls],
                            'score': int(game_state.score),
                            'game_over': game_state.game_over,
                            'path': game_state.path,
                            'difficulty': round(game_state.difficulty, 1)
                        }
                        client_socket.send(json.dumps(state).encode('utf-8'))
                        request_queue.pop(0)
                    else:
                        # 如果遇到无法处理的请求，清空队列
                        request_queue.clear()
                        break

            except socket.timeout:
                # 发送心跳包
                client_socket.send(json.dumps({'action': 'heartbeat'}).encode('utf-8'))
    except Exception as e:
        print(f"客户端连接错误: {e}")
    finally:
        client_socket.close()

# 主游戏循环
def main():
    game_state = GameState()
    clock = pygame.time.Clock()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5555))
    server_socket.listen(5)
    print("服务器启动，等待连接...")

    def accept_clients():
        while True:
            client_socket, addr = server_socket.accept()
            print(f"新连接来自: {addr}")
            client_thread = threading.Thread(target=handle_client, args=(client_socket, game_state))
            client_thread.start()

    accept_thread = threading.Thread(target=accept_clients)
    accept_thread.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game_state.update()
        game_state.draw()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    server_socket.close()

if __name__ == "__main__":
    main()
