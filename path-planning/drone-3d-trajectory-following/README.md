# 四旋翼无人机3D轨迹跟踪

## 概述

本项目实现了四旋翼无人机在3D空间中的轨迹跟踪控制仿真。代码基于五次多项式轨迹生成器，使用PD控制器实现四旋翼的姿态和位置控制。

**来源**: 本项目代码下载自 [PythonRobotics 项目](https://github.com/AtsushiSakai/PythonRobotics/tree/master/AerialNavigation/drone_3d_trajectory_following)

**作者**: Daniel Ingram (daniel-s-ingram)

## 文件结构

```
drone-3d-trajectory-following/
├── Quadrotor.py                      # 四旋翼可视化类
├── TrajectoryGenerator.py            # 轨迹生成器类
├── drone_3d_trajectory_following.py  # 主程序
├── __init__.py                       # 模块初始化文件
└── README.md                         # 说明文档
```

## 核心模块说明

### 1. Quadrotor.py - 四旋翼可视化类

负责四旋翼无人机的3D可视化展示。

**主要功能**:
- 初始化四旋翼的位置和姿态（x, y, z, roll, pitch, yaw）
- 实时更新四旋翼的位姿状态
- 使用matplotlib进行3D动画显示
- 绘制四旋翼飞行轨迹

**关键方法**:
- `__init__()`: 初始化四旋翼，设置初始位置、姿态和可视化窗口
- `update_pose()`: 更新四旋翼的位置和姿态，并记录轨迹数据
- `transformation_matrix()`: 计算从机体坐标系到世界坐标系的变换矩阵
- `plot()`: 绘制四旋翼当前状态和飞行轨迹

**可视化特点**:
- 显示四旋翼的四个旋翼位置（黑色点）
- 显示四旋翼的结构框架（红色连线）
- 显示飞行轨迹（蓝色虚线）
- 显示范围：x, y: [-5, 5]，z: [0, 10]

### 2. TrajectoryGenerator.py - 轨迹生成器

使用五次多项式生成平滑的3D轨迹。

**主要功能**:
- 生成满足起始和终止位置、速度、加速度约束的五次多项式轨迹
- 为x、y、z三个维度分别计算轨迹系数

**数学原理**:
轨迹采用五次多项式形式：
```
p(t) = c₀t⁵ + c₁t⁴ + c₂t³ + c₃t² + c₄t + c₅
```

通过求解线性方程组，满足以下边界条件：
- 起始位置、速度、加速度
- 终止位置、速度、加速度

**关键方法**:
- `__init__()`: 初始化起始和目标状态
- `solve()`: 求解轨迹系数，分别计算x、y、z三个维度的系数

**参数说明**:
- `start_pos`: 起始位置 [x, y, z]
- `des_pos`: 目标位置 [x, y, z]
- `T`: 轨迹时间
- `start_vel`: 起始速度 [vx, vy, vz]（默认[0,0,0]）
- `des_vel`: 目标速度 [vx, vy, vz]（默认[0,0,0]）
- `start_acc`: 起始加速度 [ax, ay, az]（默认[0,0,0]）
- `des_acc`: 目标加速度 [ax, ay, az]（默认[0,0,0]）

### 3. drone_3d_trajectory_following.py - 主程序

实现四旋翼轨迹跟踪控制的完整仿真。

**主要功能**:
- 生成矩形轨迹（四个航点）
- 使用PD控制器控制四旋翼跟踪轨迹
- 模拟四旋翼动力学

**控制算法**:

1. **位置控制（Z轴）**:
   ```
   thrust = m * (g + des_z_acc + Kp_z * (des_z_pos - z_pos) + Kd_z * (des_z_vel - z_vel))
   ```

2. **姿态控制（Roll/Pitch）**:
   ```
   roll_torque = Kp_roll * (((des_x_acc * sin(des_yaw) - des_y_acc * cos(des_yaw)) / g) - roll)
   pitch_torque = Kp_pitch * (((des_x_acc * cos(des_yaw) - des_y_acc * sin(des_yaw)) / g) - pitch)
   ```

3. **姿态控制（Yaw）**:
   ```
   yaw_torque = Kp_yaw * (des_yaw - yaw)
   ```

**仿真参数**:
- 重力加速度: `g = 9.81 m/s²`
- 四旋翼质量: `m = 0.2 kg`
- 转动惯量: `Ixx = Iyy = Izz = 1 kg·m²`
- 轨迹时间: `T = 5 s`
- 控制周期: `dt = 0.1 s`

**控制增益**:
- 位置增益: `Kp_x = Kp_y = Kp_z = 1`, `Kd_z = 1`, `Kd_x = Kd_y = 10`
- 姿态增益: `Kp_roll = Kp_pitch = Kp_yaw = 25`

**关键函数**:
- `main()`: 主函数，生成轨迹并启动仿真
- `quad_sim()`: 四旋翼仿真主循环
- `calculate_position()`: 根据轨迹系数计算位置
- `calculate_velocity()`: 根据轨迹系数计算速度
- `calculate_acceleration()`: 根据轨迹系数计算加速度
- `rotation_matrix()`: 计算ZYX欧拉角的旋转矩阵

**轨迹规划**:
程序预设了四个航点，形成一个矩形轨迹：
- 航点1: [-5, -5, 5]
- 航点2: [5, -5, 5]
- 航点3: [5, 5, 5]
- 航点4: [-5, 5, 5]

四旋翼会循环执行这个矩形轨迹8次。

## 使用方法

### 依赖库

```bash
pip install numpy matplotlib
```

### 运行仿真

```bash
python drone_3d_trajectory_following.py
```

### 交互操作

- 按 `ESC` 键可以退出仿真

### 关闭动画

如果想关闭可视化（仅进行数值仿真），可以修改 `drone_3d_trajectory_following.py` 中的：
```python
show_animation = False
```

## 算法特点

1. **平滑轨迹**: 使用五次多项式确保位置、速度、加速度的连续性
2. **PD控制**: 采用比例-微分控制器，实现稳定的轨迹跟踪
3. **简化模型**: 使用简化的四旋翼动力学模型，便于理解和实现
4. **实时可视化**: 提供3D可视化界面，直观展示飞行过程

## 扩展方向

1. **更复杂的轨迹**: 可以修改航点生成更复杂的3D轨迹
2. **改进控制器**: 可以尝试PID控制器或其他高级控制方法
3. **添加扰动**: 可以添加风扰动、噪声等以测试控制器的鲁棒性
4. **物理参数调优**: 可以调整质量、转动惯量等参数以匹配真实四旋翼
5. **障碍物避让**: 可以结合路径规划算法实现动态避障

## 参考资料

- [PythonRobotics GitHub 仓库](https://github.com/AtsushiSakai/PythonRobotics)
- [PythonRobotics 文档](https://atsushisakai.github.io/PythonRobotics/)

