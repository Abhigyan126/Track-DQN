import pygame
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from noise import pnoise1
import os

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800
GAME_WIDTH, GAME_HEIGHT = 2400, 1600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Enhanced Self-driving Car with Deep Q-Learning")

clock = pygame.time.Clock()

try:
    CAR_IMG = pygame.image.load("car.png")
    CAR_IMG = pygame.transform.scale(CAR_IMG, (30, 30))
except:
    CAR_IMG = None

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (34, 139, 34)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
CENTER = (GAME_WIDTH // 2, GAME_HEIGHT // 2)
RADIUS = 350
ROAD_WIDTH = 100
POINTS = 120
MAX_STEPS = 2000
RAY_LENGTH = 200

class DeepQNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class Environment:

    def __init__(self):
        self.camera_offset = [0, 0]
        self.laps_completed = 0
        self.checkpoint_radius = 80
        self.last_checkpoint_time = 0
        self.checkpoint_cooldown = 2000
        self.reset()
        
    def generate_road(self):
        outer_points = []
        inner_points = []
        base_noise_offset = random.uniform(0, 100)
        secondary_noise_offset = random.uniform(0, 100)
        variation_noise_offset = random.uniform(0, 100)
        base_radius_variation = random.uniform(0.8, 1.2)
        base_radius = RADIUS * base_radius_variation

        for i in range(POINTS + 1):
            angle = 2 * math.pi * i / POINTS
            primary_noise = pnoise1((i * 0.1 + base_noise_offset)) * 60
            secondary_noise = pnoise1((i * 0.2 + secondary_noise_offset)) * 30
            variation_noise = pnoise1((i * 0.05 + variation_noise_offset)) * 20
            combined_noise = (primary_noise +
                            secondary_noise * 0.5 +
                            variation_noise * 0.3)
            if random.random() < 0.1:
                bulge = random.uniform(-30, 30)
                combined_noise += bulge

            current_radius = base_radius + combined_noise
            current_radius = max(current_radius, ROAD_WIDTH * 2)

            outer_x = CENTER[0] + (current_radius + ROAD_WIDTH / 2) * math.cos(angle)
            outer_y = CENTER[1] + (current_radius + ROAD_WIDTH / 2) * math.sin(angle)
            outer_points.append((outer_x, outer_y))

            inner_x = CENTER[0] + (current_radius - ROAD_WIDTH / 2) * math.cos(angle)
            inner_y = CENTER[1] + (current_radius - ROAD_WIDTH / 2) * math.sin(angle)
            inner_points.append((inner_x, inner_y))

        outer_points[-1] = outer_points[0]
        inner_points[-1] = inner_points[0]

        smoothed_outer = []
        smoothed_inner = []

        for i in range(len(outer_points)):
            prev_i = (i - 1) % (len(outer_points) - 1)
            next_i = (i + 1) % (len(outer_points) - 1)

            smooth_outer_x = (outer_points[prev_i][0] + outer_points[i][0] + outer_points[next_i][0]) / 3
            smooth_outer_y = (outer_points[prev_i][1] + outer_points[i][1] + outer_points[next_i][1]) / 3
            smoothed_outer.append((smooth_outer_x, smooth_outer_y))

            smooth_inner_x = (inner_points[prev_i][0] + inner_points[i][0] + inner_points[next_i][0]) / 3
            smooth_inner_y = (inner_points[prev_i][1] + inner_points[i][1] + inner_points[next_i][1]) / 3
            smoothed_inner.append((smooth_inner_x, smooth_inner_y))

        smoothed_outer[-1] = smoothed_outer[0]
        smoothed_inner[-1] = smoothed_inner[0]

        return smoothed_outer, smoothed_inner

    def reset(self):
        self.outer_points, self.inner_points = self.generate_road()
        start_angle = random.uniform(0, 2 * math.pi)
        self.spawn_point = [
            CENTER[0] + RADIUS * math.cos(start_angle),
            CENTER[1] + RADIUS * math.sin(start_angle)
        ]
        self.car_pos = self.spawn_point.copy()
        self.car_angle = math.degrees(start_angle) + 90
        self.car_speed = 0
        self.steps = 0
        self.laps_completed = 0
        self.last_checkpoint_time = pygame.time.get_ticks()
        self.update_camera()
        return self.get_state()
    def check_checkpoint(self):
        current_time = pygame.time.get_ticks()
        distance = math.sqrt((self.car_pos[0] - self.spawn_point[0])**2 +
                           (self.car_pos[1] - self.spawn_point[1])**2)
        if (distance < self.checkpoint_radius and
            current_time - self.last_checkpoint_time > self.checkpoint_cooldown):
            self.laps_completed += 1
            self.last_checkpoint_time = current_time
            if self.laps_completed % 2 == 0:
                print("Road Generated")
                old_spawn = self.spawn_point.copy()
                self.outer_points, self.inner_points = self.generate_road()
                self.spawn_point = old_spawn
                return True
        return False
    def update_camera(self):
        target_x = WINDOW_WIDTH//2 - self.car_pos[0]
        target_y = WINDOW_HEIGHT//2 - self.car_pos[1]
        self.camera_offset[0] += (target_x - self.camera_offset[0]) * 0.1
        self.camera_offset[1] += (target_y - self.camera_offset[1]) * 0.1
    def world_to_screen(self, pos):
        return (pos[0] + self.camera_offset[0], pos[1] + self.camera_offset[1])
    def cast_rays(self):
        ray_distances = []
        ray_angles = [i * 45 for i in range(8)]
        for angle in ray_angles:
            ray_angle = math.radians(self.car_angle + angle)
            ray_end = (
                self.car_pos[0] + RAY_LENGTH * math.cos(ray_angle),
                self.car_pos[1] + RAY_LENGTH * math.sin(ray_angle)
            )
            min_distance = RAY_LENGTH
            for i in range(len(self.outer_points) - 1):
                for points in [self.outer_points, self.inner_points]:
                    start = points[i]
                    end = points[i + 1]
                    x1, y1 = self.car_pos
                    x2, y2 = ray_end
                    x3, y3 = start
                    x4, y4 = end
                    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if den != 0:
                        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            px = x1 + t * (x2 - x1)
                            py = y1 + t * (y2 - y1)
                            distance = math.sqrt((px - x1)**2 + (py - y1)**2)
                            min_distance = min(min_distance, distance)
            ray_distances.append(min_distance / RAY_LENGTH)
            if pygame.display.get_surface():
                ray_end = (
                    self.car_pos[0] + min_distance * math.cos(ray_angle),
                    self.car_pos[1] + min_distance * math.sin(ray_angle)
                )
                start_screen = self.world_to_screen(self.car_pos)
                end_screen = self.world_to_screen(ray_end)
                pygame.draw.line(screen, YELLOW, start_screen, end_screen, 2)
        return ray_distances
    def get_state(self):
        ray_distances = self.cast_rays()
        return np.array(ray_distances + [self.car_speed / 5.0])
    def step(self, action):
        if action == 1:
            self.car_speed = min(self.car_speed + 0.2, 5)
        elif action == 2:
            self.car_speed = max(self.car_speed - 0.2, -2)
        elif action == 3:
            self.car_angle -= 3 if self.car_speed > 0 else -3
        elif action == 4:
            self.car_angle += 3 if self.car_speed > 0 else -3

        self.car_speed *= 0.98
        self.car_pos[0] += self.car_speed * math.cos(math.radians(self.car_angle))
        self.car_pos[1] += self.car_speed * math.sin(math.radians(self.car_angle))
        self.update_camera()

        inside_road = self.is_inside_road(self.car_pos[0], self.car_pos[1])
        reward = 1 if inside_road else -100
        reward += self.car_speed * 0.1
        
        self.steps += 1
        done = not inside_road or self.steps >= MAX_STEPS
        
        # Get the next state
        next_state = self.get_state()

        # Check checkpoint and add reward if completed
        if self.check_checkpoint():
            reward += 100

        return next_state, reward, done
    def is_inside_road(self, x, y):
        point = pygame.math.Vector2(x, y)
        road_polygon = [pygame.math.Vector2(p) for p in (self.outer_points + self.inner_points[::-1])]
        return self.point_in_polygon(point, road_polygon)
    def point_in_polygon(self, point, polygon):
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            if ((polygon[i].y > point.y) != (polygon[j].y > point.y) and
                point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) /
                (polygon[j].y - polygon[i].y) + polygon[i].x):
                inside = not inside
            j = i
        return inside
    def render(self):
        screen.fill(GREEN)
        road_points = [self.world_to_screen(p) for p in (self.outer_points + self.inner_points[::-1])]
        pygame.draw.polygon(screen, GRAY, road_points)
        pygame.draw.lines(screen, DARK_GRAY, True, [self.world_to_screen(p) for p in self.outer_points], 3)
        pygame.draw.lines(screen, DARK_GRAY, True, [self.world_to_screen(p) for p in self.inner_points], 3)
        spawn_screen = self.world_to_screen(self.spawn_point)
        pygame.draw.circle(screen, RED, spawn_screen, self.checkpoint_radius, 3)
        pygame.draw.line(screen, RED,
                        (spawn_screen[0] - self.checkpoint_radius, spawn_screen[1]),
                        (spawn_screen[0] + self.checkpoint_radius, spawn_screen[1]), 3)
        car_pos_screen = self.world_to_screen(self.car_pos)
        if CAR_IMG:
            rotated_car = pygame.transform.rotate(CAR_IMG, -self.car_angle)
            car_rect = rotated_car.get_rect(center=car_pos_screen)
            screen.blit(rotated_car, car_rect)
        else:
            car_points = [
                (car_pos_screen[0] + 15 * math.cos(math.radians(self.car_angle)),
                 car_pos_screen[1] + 15 * math.sin(math.radians(self.car_angle))),
                (car_pos_screen[0] + 8 * math.cos(math.radians(self.car_angle + 140)),
                 car_pos_screen[1] + 8 * math.sin(math.radians(self.car_angle + 140))),
                (car_pos_screen[0] + 8 * math.cos(math.radians(self.car_angle - 140)),
                 car_pos_screen[1] + 8 * math.sin(math.radians(self.car_angle - 140)))
            ]
            pygame.draw.polygon(screen, RED, car_points)
        font = pygame.font.Font(None, 36)
        speed_text = f"Speed: {abs(self.car_speed):.1f}"
        lap_text = f"Laps: {self.laps_completed}"
        speed_surface = font.render(speed_text, True, WHITE)
        lap_surface = font.render(lap_text, True, WHITE)
        screen.blit(speed_surface, (10, 10))
        screen.blit(lap_surface, (10, 50))
        pygame.display.flip()
def load_model(filename):
    if os.path.exists(filename):
        model = DeepQNetwork(9, 5)
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model
    return None
def train(continue_training=False):
    env = Environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if continue_training and os.path.exists("car_model.pth"):
        policy_net = load_model("car_model.pth").to(device)
        print("Loaded existing model")
    else:
        policy_net = DeepQNetwork(9, 5).to(device)
        print("Created new model")
    target_net = DeepQNetwork(9, 5).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(50000)
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 2000
    TARGET_UPDATE = 10
    steps_done = 0
    episode_rewards = []
    best_reward = float('-inf')
    num_episodes = 1000
    try:
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            for t in range(MAX_STEPS):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
                steps_done += 1
                if random.random() > eps_threshold:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        action = policy_net(state_tensor).max(1)[1].item()
                else:
                    action = random.randrange(5)
                next_state, reward, done = env.step(action)
                total_reward += reward
                memory.push(state, action, reward, next_state, done)
                state = next_state
                env.render()
                clock.tick(60)
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch = list(zip(*transitions))
                    state_batch = torch.FloatTensor(batch[0]).to(device)
                    action_batch = torch.LongTensor(batch[1]).to(device)
                    reward_batch = torch.FloatTensor(batch[2]).to(device)
                    next_state_batch = torch.FloatTensor(batch[3]).to(device)
                    done_batch = torch.FloatTensor(batch[4]).to(device)
                    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
                    next_state_values = target_net(next_state_batch).max(1)[0].detach()
                    expected_state_action_values = reward_batch + GAMMA * next_state_values * (1 - done_batch)
                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if done:
                    break
            episode_rewards.append(total_reward)
            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(policy_net.state_dict(), "car_model_best.pth")
            if episode % 10 == 0:
                torch.save(policy_net.state_dict(), "car_model.pth")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Best Average Reward: {best_reward:.2f}")
            print(f"Exploration Rate: {eps_threshold:.2f}")
            print("-" * 50)
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        torch.save(policy_net.state_dict(), "car_model_final.pth")
        print("Final model saved")
    return policy_net
def run_trained_model():
    env = Environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("car_model_best.pth")
    if model is None:
        print("No trained model found!")
        return
    model = model.to(device)
    model.eval()
    try:
        while True:
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = model(state_tensor).max(1)[1].item()
                state, reward, done = env.step(action)
                total_reward += reward
                env.render()
                clock.tick(60)
            print(f"Episode finished with reward: {total_reward:.2f}")
    except KeyboardInterrupt:
        print("\nSimulation ended by user")
if __name__ == "__main__":
    print("Select mode:")
    print("1. Train new model")
    print("2. Continue training existing model")
    print("3. Run trained model")
    while True:
        try:
            choice = int(input("Enter your choice (1-3): "))
            if choice in [1, 2, 3]:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    if choice == 1:
        model = train(continue_training=False)
    elif choice == 2:
        model = train(continue_training=True)
    else:
        run_trained_model()
    pygame.quit()
