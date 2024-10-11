import pygame
import socket
import json
import threading
import math
import time

# 初始化Pygame
pygame.init()

# 设置窗口大小
WIDTH = 1024
HEIGHT = 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eddie's Zuma Game Client")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

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

client_socket = connect_to_server()

# 游戏状态
game_state = None

# 获取游戏状态的函数
def get_game_state():
    global game_state, client_socket
    while True:
        try:
            client_socket.send(json.dumps({'action': 'get_state'}).encode('utf-8'))
            data = b''
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        raise ConnectionError("Connection closed by server")
                    data += chunk
                    if len(chunk) < 4096:
                        break
                except socket.timeout:
                    print("接收数据超时，重试中...")
                    continue
                except ConnectionError as e:
                    print(f"连接错误: {e}")
                    client_socket = connect_to_server()
                    break
            try:
                game_state = json.loads(data.decode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"接收到的数据: {data.decode('utf-8')}")
        except Exception as e:
            print(f"获取游戏状态失败: {e}")
            client_socket = connect_to_server()

# 开始获取游戏状态的线程
state_thread = threading.Thread(target=get_game_state)
state_thread.daemon = True
state_thread.start()

# 发送操作到服务器
def send_action(action, direction=None):
    global client_socket
    try:
        if direction is not None:
            client_socket.send(json.dumps({'action': action, 'direction': direction}).encode('utf-8'))
        else:
            client_socket.send(json.dumps({'action': action}).encode('utf-8'))
    except Exception as e:
        print(f"发送操作失败: {e}")
        client_socket = connect_to_server()

# 主游戏循环
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                send_action('rotate', -1)
            elif event.key == pygame.K_RIGHT:
                send_action('rotate', 1)
            elif event.key == pygame.K_SPACE:
                send_action('shoot')

    # 绘制游戏状态
    if game_state:
        screen.fill(WHITE)
        # 绘制路径
        if 'path' in game_state:
            pygame.draw.lines(screen, (200, 200, 200), False, game_state['path'], 2)
        
        # 绘制发射器
        if 'shooter_angle' in game_state and 'shooter_color' in game_state:
            shooter_pos = (WIDTH // 2, HEIGHT // 2)
            pygame.draw.circle(screen, game_state['shooter_color'], shooter_pos, 30)
            end_x = shooter_pos[0] + 50 * math.cos(math.radians(game_state['shooter_angle']))
            end_y = shooter_pos[1] - 50 * math.sin(math.radians(game_state['shooter_angle']))
            pygame.draw.line(screen, BLACK, shooter_pos, (int(end_x), int(end_y)), 5)
        
        # 绘制球
        if 'balls' in game_state:
            for ball_pos, ball_color in game_state['balls']:
                pygame.draw.circle(screen, ball_color, (int(ball_pos[0]), int(ball_pos[1])), 20)
        
        # 绘制分数
        if 'score' in game_state:
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {game_state['score']}", True, BLACK)
            screen.blit(score_text, (10, 10))
        
        # 绘制难度
        if 'difficulty' in game_state:
            font = pygame.font.Font(None, 36)
            difficulty_text = font.render(f"Difficulty: {game_state['difficulty']:.1f}", True, BLACK)
            screen.blit(difficulty_text, (10, 50))
        
        # 绘制游戏结束信息
        if 'game_over' in game_state and game_state['game_over']:
            font = pygame.font.Font(None, 72)
            game_over_text = font.render("Game Over", True, (255, 0, 0))
            screen.blit(game_over_text, (WIDTH//2 - 100, HEIGHT//2))
    else:
        screen.fill(WHITE)
        font = pygame.font.Font(None, 36)
        waiting_text = font.render("Waiting for game state...", True, BLACK)
        screen.blit(waiting_text, (WIDTH//2 - 100, HEIGHT//2))

    pygame.display.flip()
    clock.tick(60)

# 关闭连接和退出
try:
    client_socket.close()
except Exception as e:
    print(f"关闭socket时发生错误: {e}")
pygame.quit()
