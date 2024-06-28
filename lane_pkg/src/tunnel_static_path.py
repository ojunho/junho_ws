import matplotlib.pyplot as plt

import numpy as np

# 현재 차량의 위치 (x, y)와 각도 (theta)
current_position = np.array([0.0, 0.0])
current_theta = 0.0

# 목표 위치 (3.8m 왼쪽 차선 중앙점)
target_position = np.array([-3.8, 10.0])

def plan_path(current_position, target_position, steps=100):
    # 직선 경로를 생성
    path = np.linspace(current_position, target_position, steps)
    return path

# 경로 계획
path = plan_path(current_position, target_position)


# 제어 변수
k_p = 1.0  # 비례 상수
k_d = 0.1  # 미분 상수

# 초기 상태
current_velocity = 0.0
current_steering_angle = 0.0

# 제어 함수
def control(current_position, target_position, current_velocity, k_p, k_d):
    error = target_position - current_position
    control_signal = k_p * error - k_d * current_velocity
    return control_signal

# 시뮬레이션
positions = [current_position]
for point in path:
    control_signal = control(current_position, point, current_velocity, k_p, k_d)
    current_velocity += control_signal
    current_position += current_velocity
    positions.append(current_position)

positions = np.array(positions)

# 경로 시각화
plt.plot(path[:, 0], path[:, 1], 'r--', label='Planned Path')
plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Vehicle Path')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Path Following Control')
plt.show()
