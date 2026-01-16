import numpy as np
from numpy.random import uniform, randn, random


class RealTarget:
    """
    真实目标模型：'匀速转弯 (CT) 模型'
    """
    def __init__(self, ID, priority, start_belief_pos, initial_v, initial_phi):
        self.ID = ID
        self.priority = priority

        self.state = np.zeros(6)  # 状态向量: [x, y, vx, vy, phi, w]
        self.state[0:2] = start_belief_pos
        self.state[2] = initial_v * np.cos(np.radians(initial_phi))
        self.state[3] = initial_v * np.sin(np.radians(initial_phi))
        self.state[4] = np.radians(initial_phi)
        self.state[5] = np.radians(5)

        self.dt = 1.0
        self.time_step = 0

        # 运动参数限制
        self.v_range = [15.0, 60.0]
        self.turn_rate = np.radians(5.0)  # 转弯速率：5度/秒
        self.phi_range = [np.radians(15), np.radians(60)]

    def step_forward(self):
        """真实目标步进"""
        if self.time_step == 30:
            self.state[5] = -np.radians(5)

        if abs(self.state[5]) > 1e-5:
            # 转弯
            omega_dt = self.state[5] * self.dt
            sin_wt = np.sin(omega_dt)
            cos_wt = np.cos(omega_dt)
            vx_prev = self.state[2]
            vy_prev = self.state[3]
            self.state[0] = self.state[0] + (vx_prev * sin_wt - vy_prev * (1 - cos_wt)) / self.state[5]
            self.state[1] = self.state[1] + (vx_prev * (1 - cos_wt) + vy_prev * sin_wt) / self.state[5]
            self.state[2] = vx_prev * cos_wt - vy_prev * sin_wt
            self.state[3] = vx_prev * sin_wt + vy_prev * cos_wt
            self.state[4] = self.state[4] + self.state[5] * self.dt  # phi = phi0 + wt
        else:
            # 直线
            self.state[0] = self.state[0] + self.state[2] * self.dt
            self.state[1] = self.state[1] + self.state[3] * self.dt


class TargetPredictor:
    """
    目标预测器：集成GWO-PF(灰狼优化粒子滤波)
    """
    def __init__(self, map_size, obstacles_map, v_range, theta_range, init_pos, num_particles=1000,
                 ID=1, priority=1, dt=1.0):
        """
        初始化目标粒子滤波预测
        :param map_size: (M,N),地图大小
        :param obstacles_map: 障碍物坐标
        :param init_pos: 初始位置
        :param num_particles:
        :param ID:
        :param priority:
        :param dt:
        """

        self.ID = ID
        self.priority = priority
        self.M, self.map_width = map_size
        self.num_particles = num_particles
        # 地图处理
        self.map = np.zeros((self.M, self.map_width))
        if obstacles_map is not None:
            obstaces_xy = np.array(obstacles_map)
            if obstaces_xy.size == 0:
                obstaces_xy = obstaces_xy.reshape(0, 2)
            else:
                obstaces_xy = obstaces_xy.astype(int, copy=False)
            # 简单的边界检查防止索引报错
            valid_obs = (obstaces_xy[:, 0] < self.M) & (obstaces_xy[:, 1] < self.map_width)
            obstaces_xy = obstaces_xy[valid_obs]
            if len(obstaces_xy) > 0:
                self.map[obstaces_xy[:, 0], obstaces_xy[:, 1]] = -1

        self.dt = dt
        self.v_min, self.v_max = v_range
        self.th_min, self.th_max = theta_range

        # 状态向量: [x, y, vx, vy, omega (转弯率)]
        # 使用 Nx5 的矩阵进行向量化操作，避免慢速循环
        self.particles = np.zeros((self.num_particles, 5))
        self.weights = np.ones(self.num_particles) / self.num_particles

        # 过程噪声标准差 (基准值)
        # [x, y, vx, vy, omega] 的扰动
        self.Q_std = np.array([0.1, 0.1, 0.5, 0.5, 0.01])

        # 扇区均匀初始化
        self._sector_initialization(init_pos)

        # 引入一个变量来存储平滑后的误差，避免Q值突变
        self.avg_innovation = 0.0
        # 限制最大自适应增益，防止噪声爆炸
        self.max_scale = 3.0

        self.meas_noise_std = 3.0
        self.state_si = 0.0

        self.alpha = 1.0
        self.k_max = 2

    def _sector_initialization(self, init_pos):
        """
        在给定的速度和角度区间内进行均匀采样。
        """
        # 位置初始化
        self.particles[:, 0] = init_pos[0] + randn(self.num_particles) * 2.0
        self.particles[:, 1] = init_pos[1] + randn(self.num_particles) * 2.0

        # 速度模长采样 (校正面密度)
        # u ~ U -> v = sqrt(v_min^2 + u*(v_max^2 - v_min^2))
        u = random(self.num_particles)
        v_sq_samples = self.v_min ** 2 + u * (self.v_max ** 2 - self.v_min ** 2)
        v_samples = np.sqrt(v_sq_samples)

        # 航向角采样
        th_samples = uniform(self.th_min, self.th_max, self.num_particles)

        # 转换回笛卡尔坐标系
        self.particles[:, 2] = v_samples * np.cos(th_samples)
        self.particles[:, 3] = v_samples * np.sin(th_samples)

        # 转弯率初始化
        self.particles[:, 4] = uniform(-np.deg2rad(5), np.deg2rad(5), self.num_particles)

    def predict(self, innovation_norm=0.0):
        """
        状态预测步骤 (向量化实现协同转弯 CT 模型)
        """
        dt = self.dt
        N = self.num_particles

        # 自适应噪声调节
        # 新息(误差)越大，说明目标可能在机动，需增大过程噪声Q
        alpha_smooth = 0.1
        self.avg_innovation = (1 - alpha_smooth) * self.avg_innovation + alpha_smooth * innovation_norm
        raw_scale = 1.0 + 4.0 * (1.0 - np.exp(-0.05 * self.avg_innovation))  # 似然函数
        alpha = min(raw_scale, self.max_scale)
        Q_curr = self.Q_std * alpha

        # 提取当前状态列
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        vx = self.particles[:, 2]
        vy = self.particles[:, 3]
        w = self.particles[:, 4]

        # 处理转弯率接近0的情况
        # 如果 w 很小，退化为匀速直线运动 (CV) 模型
        # 使用 np.where 进行向量化分支处理
        small_w = np.abs(w) < 1e-4

        # CT 模型更新方程
        # 速度更新
        # vx' = vx*cos(wt) - vy*sin(wt)
        # vy' = vx*sin(wt) + vy*cos(wt)
        sin_wt = np.sin(w * dt)
        cos_wt = np.cos(w * dt)

        new_vx = vx * cos_wt - vy * sin_wt
        new_vy = vx * sin_wt + vy * cos_wt

        # 位置更新
        # CT模型积分形式
        # dx = (vx*sin(wt) - vy*(1-cos(wt))) / w
        # dy = (vx*(1-cos(wt)) + vy*sin(wt)) / w

        # 对于非零 w:
        dx_ct = (vx * sin_wt - vy * (1 - cos_wt)) / w
        dy_ct = (vx * (1 - cos_wt) + vy * sin_wt) / w

        # 对于接近零 w (CV模型近似): dx = vx*dt, dy = vy*dt
        dx_cv = vx * dt
        dy_cv = vy * dt

        # 合并两种情况
        dx = np.where(small_w, dx_cv, dx_ct)
        dy = np.where(small_w, dy_cv, dy_ct)

        new_x = x + dx
        new_y = y + dy
        new_w = w

        # 将更新后的无噪声状态写回
        self.particles[:, 0] = new_x
        self.particles[:, 1] = new_y
        self.particles[:, 2] = new_vx
        self.particles[:, 3] = new_vy
        self.particles[:, 4] = new_w

        # 添加过程噪声
        noise = randn(N, 5) * Q_curr
        # 位置噪声通常较小，速度和转弯率噪声主要驱动机动
        # 这里给位置加一点小噪声防止粒子重叠
        self.particles += noise

    def update(self, z, R_cov, det_res, min_val=0.001):
        """
        量测更新与软约束权重惩罚
        """
        # 计算似然
        likelihood = np.ones(self.num_particles)
        if z is not None:
            # Case 1: 探测到目标
            # z: [x, y]
            dx = z[0] - self.particles[:, 0]
            dy = z[1] - self.particles[:, 1]
            dist_sq = dx ** 2 + dy ** 2
            # 似然 P(z|x)
            likelihood = np.exp(-dist_sq / (2 * R_cov))

        else:
            # Case 2: 未探测到目标
            # 如果粒子在任意无人机的探测范围内，但无人机没看到，那么该粒子的权重应该降低
            sensor_r = 250.0  # 探测半径
            sensor_r_sq = sensor_r ** 2
            coverage_counts = np.zeros(self.num_particles)  # 计入粒子被几个无人机惩罚过
            neg_likelihood = np.ones(self.num_particles)
            # 对于每一架没看到目标的无人机
            for det in det_res:
                if not det["detected"]:
                    # 计算粒子到该无人机的距离
                    p_dx = self.particles[:, 0] - det["uavpos"][0]
                    p_dy = self.particles[:, 1] - det["uavpos"][1]
                    pd = det["uavdp"]  # 无人机检测概率
                    p_dist_sq = p_dx ** 2 + p_dy ** 2

                    # 找出在探测范围内的粒子
                    in_range_mask = p_dist_sq < sensor_r_sq
                    valid_mask = in_range_mask & (coverage_counts < self.k_max)

                    if np.any(valid_mask):
                        neg_likelihood[valid_mask] *= (1.0 - pd)
                        coverage_counts[in_range_mask] += 1

            neg_likelihood = np.power(neg_likelihood, self.alpha)
            neg_likelihood = np.maximum(neg_likelihood, min_val)
            likelihood *= neg_likelihood

        # 软约束
        # 计算每个粒子的速度和航向
        v_curr = np.sqrt(self.particles[:, 2] ** 2 + self.particles[:, 3] ** 2)
        th_curr = np.arctan2(self.particles[:, 3], self.particles[:, 2])

        k = 10.0  # 硬度系数

        # 速度约束
        pen_v_min = 1.0 / (1.0 + np.exp(k * (self.v_min - v_curr)))
        pen_v_max = 1.0 / (1.0 + np.exp(k * (v_curr - self.v_max)))

        # 角度约束
        pen_th_min = 1.0 / (1.0 + np.exp(k * (self.th_min - th_curr)))
        pen_th_max = 1.0 / (1.0 + np.exp(k * (th_curr - self.th_max)))

        penalty = pen_v_min * pen_v_max * pen_th_min * pen_th_max

        # --- C. 综合更新 ---
        # w = w * likelihood * penalty
        self.weights *= (likelihood * penalty) + 1e-300
        # 归一化
        w_sum = np.sum(self.weights)
        if w_sum == 0:
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= w_sum

    def estimate(self):
        """计算加权均值作为状态估计"""
        return np.average(self.particles, weights=self.weights, axis=0)

    def resample(self):
        """系统重采样"""
        # 有效粒子数
        N_eff = 1.0 / np.sum(self.weights ** 2)

        # 如果有效粒子太少，则重采样
        if N_eff < self.num_particles / 2:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)

    def step_update(self, z, innovation_norm, det_res):
        self.predict(innovation_norm=innovation_norm)
        self.update(z, R_cov=self.meas_noise_std ** 2, det_res=det_res)
        self.state_si = self.estimate()  # 加权均值，用作预测RSME误差计算
        self.resample()  #  重采样

        return self._generate_grid()

    def get_entropy(self):
        """
        计算香农熵 (用于奖励函数)
        """
        # 过滤掉极小权重以避免 log(0)
        valid_weights = self.weights[self.weights > 1e-12]
        if len(valid_weights) == 0:
            return 0.0

        # 归一化
        valid_weights /= np.sum(valid_weights)

        # H = -sum(p * log(p))
        return -np.sum(valid_weights * np.log(valid_weights))

    def get_local_entropy(self, center, radius):
        """
        计算局部熵（只关注无人机附近的粒子分布）。
        """
        d_vec = self.particles[:, 0:2] - center
        dists = np.linalg.norm(d_vec, axis=1)
        mask = dists <= radius
        if not np.any(mask):
            return 0.0

        local_weights = self.weights[mask]
        local_weights = local_weights[local_weights > 1e-12]
        if len(local_weights) == 0:
            return 0.0

        local_weights = local_weights / np.sum(local_weights)
        return -np.sum(local_weights * np.log(local_weights))

    def _generate_grid(self):
        """生成概率网格 """
        px = self.particles[:, 0]
        py = self.particles[:, 1]
        col = np.floor(px).astype(int)
        row = (self.M - 1) - np.floor(py).astype(int)

        valid = (row >= 0) & (row < self.M) & (col >= 0) & (col < self.map_width)

        prob_grid = np.zeros((self.M, self.map_width))
        if np.any(valid):
            # 使用加权直方图
            H, _, _ = np.histogram2d(
                row[valid], col[valid],
                bins=[np.arange(self.M + 1), np.arange(self.map_width + 1)],
                weights=self.weights[valid],  # 关键：使用权重
                density=False
            )
            prob_grid = H

        # 再次归一化确保输出是概率分布
        total = np.sum(prob_grid)
        if total > 0:
            prob_grid /= total

        return prob_grid
