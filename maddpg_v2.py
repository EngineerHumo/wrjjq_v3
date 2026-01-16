import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 多智能体经验回放池 (Global Buffer) ---
class MultiAgentReplayBuffer:
    def __init__(self, max_size, num_agents, state_dims, action_dims):
        """
        :param max_size: 经验池最大容量
        :param num_agents: 智能体数量
        :param state_dims: list, 每个智能体的状态维度 [dim1, dim2, ...]
        :param action_dims: list, 每个智能体的动作维度 [dim1, dim2, ...]
        """
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.num_agents = num_agents

        # 初始化存储容器：self.obs_n[i] 存储第 i 个智能体的观测历史
        self.obs_n = [np.zeros((self.max_size, state_dims[i])) for i in range(num_agents)]
        self.act_n = [np.zeros((self.max_size, action_dims[i])) for i in range(num_agents)]
        self.rew_n = np.zeros((self.max_size, num_agents))  # 奖励矩阵 [size, num_agents]
        self.next_obs_n = [np.zeros((self.max_size, state_dims[i])) for i in range(num_agents)]
        self.done_n = np.zeros((self.max_size, num_agents))

    def add(self, obs_list, act_list, rew_list, next_obs_list, done_list):
        """添加一条多智能体联合样本"""
        idx = self.ptr
        for i in range(self.num_agents):
            self.obs_n[i][idx] = obs_list[i]
            self.act_n[i][idx] = act_list[i]
            self.next_obs_n[i][idx] = next_obs_list[i]

        self.rew_n[idx] = np.array(rew_list)
        self.done_n[idx] = np.array(done_list)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """随机采样"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch_obs_n = [torch.FloatTensor(self.obs_n[i][idxs]).to(device) for i in range(self.num_agents)]
        batch_act_n = [torch.FloatTensor(self.act_n[i][idxs]).to(device) for i in range(self.num_agents)]
        batch_rew_n = torch.FloatTensor(self.rew_n[idxs]).to(device)
        batch_next_obs_n = [torch.FloatTensor(self.next_obs_n[i][idxs]).to(device) for i in range(self.num_agents)]
        batch_done_n = torch.FloatTensor(self.done_n[idxs]).to(device)

        return batch_obs_n, batch_act_n, batch_rew_n, batch_next_obs_n, batch_done_n


# --- 2. 神经网络定义 ---
class Actor(nn.Module):
    """演员网络：只输入自己的 state"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """评论家网络 (Global Critic)：输入所有人的 states 和 actions"""
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()
        # 输入维度 = sum(all_states) + sum(all_actions)
        self.l1 = nn.Linear(total_state_dim + total_action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state_cat, action_cat):
        x = torch.cat([state_cat, action_cat], dim=1)
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        return self.l3(q)


# --- 3. MADDPG 核心结构 ---
class MADDPG:
    """每个智能体持有的算法核心（网络 + 优化器）"""
    def __init__(self, state_dim, action_dim, max_action, total_state_dim, total_action_dim,
                 actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99):
        # Local Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Global Critic
        self.critic = Critic(total_state_dim, total_action_dim).to(device)
        self.critic_target = Critic(total_state_dim, total_action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = gamma
        self.tau = tau  # 软更新系数

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_tensor = self.actor(state_tensor).detach().cpu()
        return action_tensor.numpy().flatten()


# --- 4. 集中式训练函数 (Centralized Training) ---
def train_centralized(agents, buffer, batch_size=64, gamma=0.99):
    """
    对所有智能体执行一步 MADDPG 更新
    """
    if buffer.size < batch_size:
        return

    # 1. 从全局 Buffer 采样
    # obs_n: list of tensors, shape [batch, state_dim]
    obs_n, act_n, rew_n, next_obs_n, done_n = buffer.sample(batch_size)

    # 2. 拼接全局向量 (for Critic)
    obs_cat = torch.cat(obs_n, dim=1)  # [batch, total_state_dim]
    act_cat = torch.cat(act_n, dim=1)  # [batch, total_action_dim]
    next_obs_cat = torch.cat(next_obs_n, dim=1)

    # 3. 遍历每个智能体进行更新
    for i, agent in enumerate(agents):
        brain = agent.brain

        # --- Update Critic ---
        with torch.no_grad():
            # 计算所有智能体在下一时刻的目标动作
            target_actions = []
            for j, other_agent in enumerate(agents):
                # 注意：使用每个 Agent 自己的 Target Actor
                target_actions.append(other_agent.brain.actor_target(next_obs_n[j]))
            target_act_cat = torch.cat(target_actions, dim=1)

            # 计算 Target Q (使用当前 Agent 的 Target Critic)
            target_Q = brain.critic_target(next_obs_cat, target_act_cat)
            target_Q = rew_n[:, i].view(-1, 1) + (1 - done_n[:, i].view(-1, 1)) * gamma * target_Q

        # Current Q
        current_Q = brain.critic(obs_cat, act_cat)

        # Loss & Step
        critic_loss = F.mse_loss(current_Q, target_Q)
        brain.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.critic.parameters(), 0.5)  # 梯度裁剪
        brain.critic_optimizer.step()

        # --- Update Actor ---
        # 计算当前动作组合 (用于评估 Actor Performance)
        # 技巧：只有当前 Agent i 的动作需要梯度，其他 Agent 的动作视为环境常量
        curr_actions = []
        for j, other_agent in enumerate(agents):
            action_j = other_agent.brain.actor(obs_n[j])
            if i != j:
                action_j = action_j.detach()  # 其他智能体的动作视为环境一部分，不传梯度
            curr_actions.append(action_j)

        curr_act_cat = torch.cat(curr_actions, dim=1)

        # Actor Loss: 最大化 Critic 的评分
        actor_loss = -brain.critic(obs_cat, curr_act_cat).mean()

        brain.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.actor.parameters(), 0.5)  # 梯度裁剪
        brain.actor_optimizer.step()

        # --- Soft Update ---
        for param, target_param in zip(brain.critic.parameters(), brain.critic_target.parameters()):
            target_param.data.copy_(brain.tau * param.data + (1 - brain.tau) * target_param.data)

        for param, target_param in zip(brain.actor.parameters(), brain.actor_target.parameters()):
            target_param.data.copy_(brain.tau * param.data + (1 - brain.tau) * target_param.data)


# --- 5. UAV Agent 类 ---
class UAVAgent:
    def __init__(self, uav_id, initial_pos, initial_v, initial_phi, step,
                 total_state_dim=None, total_action_dim=None,
                 actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99):
        # 物理属性
        self.step = step
        self.ID = uav_id
        self.pos = np.array(initial_pos, dtype=float)
        self.v = float(initial_v)
        self.phi = np.radians(initial_phi)

        # 约束
        self.velocity_range = [28.0, 70.0]
        self.acc_range = [-8, 8]
        self.w_range = [-np.radians(5), np.radians(5)]
        self.comm_range = 60.0
        self.detect_radius = 250.0  # 无人机探测半径
        self.detecct_p = 0.9

        # 状态与动作维度
        self.state_dim = 17
        self.action_dim = 2
        self.max_action_tensor = torch.tensor([1.0, 1.0]).to(device)

        # 初始化 MADDPG
        if total_state_dim is None:
            total_state_dim = self.state_dim
        if total_action_dim is None:
            total_action_dim = self.action_dim

        self.brain = MADDPG(self.state_dim, self.action_dim, self.max_action_tensor,
                            total_state_dim, total_action_dim,
                            actor_lr, critic_lr, tau, gamma)

        self.assigned_task_coords = None

    @staticmethod
    def _compute_global_belief_direction(agent_pos, map_size, target_predictors):
        if not target_predictors:
            return 0.0, 0.0

        weighted_sum = np.zeros(2)
        total_weight = 0.0
        for predictor in target_predictors:
            estimate = predictor.estimate()[0:2]
            weight = float(getattr(predictor, "priority", 1.0))
            weighted_sum += estimate * weight
            total_weight += weight

        if total_weight <= 1e-6:
            return 0.0, 0.0

        global_point = weighted_sum / total_weight
        dx, dy = global_point[0] - agent_pos[0], global_point[1] - agent_pos[1]
        norm = np.hypot(dx, dy)
        if norm <= 1e-6:
            return 0.0, 0.0

        return dx / norm, dy / norm

    def get_observation(self, map_size, obstacles_map, all_uavs, target_predictors):
        """
        获取观测状态
        """
        M, N = map_size
        x, y = self.pos

        # --- 1. 处理通信网络 (邻居) ---
        neighbors = []
        for other in all_uavs:
            if other is self:
                continue
            d_vec = other.pos - self.pos
            if np.linalg.norm(d_vec) <= self.comm_range:
                neighbors.append(other)

        num_neighbors = len(neighbors)
        has_neighbor = 0.0
        n_dx, n_dy = 0.0, 0.0

        if num_neighbors > 0:
            closest_uav = min(neighbors, key=lambda u: np.linalg.norm(u.pos - self.pos))
            n_dx = closest_uav.pos[0] - x
            n_dy = closest_uav.pos[1] - y
            has_neighbor = 1.0

        # --- 2. 处理协同障碍物 ---
        def scan_obstacles(agent_pos):
            detected = []
            if obstacles_map is None:
                return []
            r_int = int(self.detect_radius)
            c_center = int(agent_pos[0])
            r_center = int((M - 1) - agent_pos[1])
            r_min = max(0, r_center - r_int)
            r_max = min(M, r_center + r_int + 1)
            c_min = max(0, c_center - r_int)
            c_max = min(N, c_center + r_int + 1)
            local_map = obstacles_map[r_min:r_max, c_min:c_max]
            obs_indices = np.argwhere(local_map == -1)
            for idx in obs_indices:
                row_global = r_min + idx[0]
                col_global = c_min + idx[1]
                obs_x = col_global + 0.5
                obs_y = (M - 1) - row_global + 0.5
                if (obs_x - agent_pos[0]) ** 2 + (obs_y - agent_pos[1]) ** 2 <= self.detect_radius ** 2:
                    detected.append(np.array([obs_x, obs_y]))
            return detected

        all_known = scan_obstacles(self.pos)
        for n_uav in neighbors:
            all_known.extend(scan_obstacles(n_uav.pos))

        has_obstacle = 0.0
        o_dx, o_dy = 0.0, 0.0

        if len(all_known) > 0:
            dists = [np.linalg.norm(op - self.pos) for op in all_known]
            min_idx = np.argmin(dists)
            nearest_obs = all_known[min_idx]
            o_dx = nearest_obs[0] - x
            o_dy = nearest_obs[1] - y
            has_obstacle = 1.0

        # --- 3. 组装 Observation ---
        obs = np.array([
            x / N, y / M,
            (self.v - self.velocity_range[0]) / (self.velocity_range[1] - self.velocity_range[0]),  # 归一化
            self.phi / np.pi,
            num_neighbors / 5.0,
            has_neighbor,
            n_dx / self.comm_range,
            n_dy / self.comm_range,
            has_obstacle,
            o_dx / self.detect_radius,
            o_dy / self.detect_radius
        ])

        # 全局信念方向
        g_dx, g_dy = self._compute_global_belief_direction(self.pos, map_size, target_predictors)
        global_belief = np.array([g_dx, g_dy])

        # 不确定性扇区感知
        uncertainty_sectors = np.zeros(4)  # [前, 左, 后, 右]

        sense_radius = self.detect_radius  # 250.0

        for predictor in target_predictors:
            # 1. 计算相对距离
            d_vec = predictor.particles[:, 0:2] - self.pos
            dists = np.linalg.norm(d_vec, axis=1)

            # 2. 只统计探测半径内的粒子
            mask = dists < sense_radius
            if not np.any(mask):
                continue

            # 3. 计算相对角度
            rel_angles = np.arctan2(d_vec[mask, 1], d_vec[mask, 0]) - self.phi
            rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

            weights = predictor.weights[mask]

            # 4. 统计扇区
            # 前 (-45 ~ 45)
            uncertainty_sectors[0] += np.sum(weights[(rel_angles >= -np.pi / 4) & (rel_angles < np.pi / 4)])
            # 左 (45 ~ 135)
            uncertainty_sectors[1] += np.sum(weights[(rel_angles >= np.pi / 4) & (rel_angles < 3 * np.pi / 4)])
            # 后 (135 ~ -135)
            uncertainty_sectors[2] += np.sum(weights[(rel_angles >= 3 * np.pi / 4) | (rel_angles < -3 * np.pi / 4)])
            # 右 (-135 ~ -45)
            uncertainty_sectors[3] += np.sum(weights[(rel_angles >= -3 * np.pi / 4) & (rel_angles < -np.pi / 4)])

        # 归一化：为了让数值匹配神经网络输入范围，稍微缩放一下
        uncertainty_sectors = np.clip(uncertainty_sectors * 5.0, 0, 5.0)

        # 拼接
        final_obs = np.concatenate((obs, global_belief, uncertainty_sectors))

        return final_obs

    def state_update(self, action, map_size, obstacles_map):
        """
        物理运动学更新 + 边界/障碍物处理
        :param action: 归一化动作 [-1, 1] -> [v_acc, w_acc] 或直接映射
        :param map_size: (M, N)
        :param obstacles_map: 障碍物矩阵
        """
        M, N = map_size

        # 1. 解析动作 (Action -> Physics)
        # 假设 action[0] 控制加速度, action[1] 控制角速度
        # 映射到物理范围: action [-1, 1] -> acc_range, w_range
        acc = action[0] * self.acc_range[1]
        w = action[1] * self.w_range[1]

        # 2. 更新速度与航向
        self.v += acc * self.step
        self.phi += w * self.step

        # 速度限幅
        self.v = np.clip(self.v, self.velocity_range[0], self.velocity_range[1])
        # 角度归一化 (-pi, pi)
        self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi

        # 3. 试探性更新位置
        dx = self.v * np.cos(self.phi) * self.step
        dy = self.v * np.sin(self.phi) * self.step

        next_x = self.pos[0] + dx
        next_y = self.pos[1] + dy

        # 4. 边界处理 (反弹)
        # 如果撞墙，不仅位置限制，角度也要反射
        if next_x < 0 or next_x >= N:
            self.phi = np.pi - self.phi  # 水平镜像反弹
            next_x = np.clip(next_x, 0, N - 0.1)

        if next_y < 0 or next_y >= M:
            self.phi = -self.phi  # 垂直镜像反弹
            next_y = np.clip(next_y, 0, M - 0.1)

        # 5. 障碍物处理
        if obstacles_map is not None:
            # 坐标转网格索引 (注意 y 轴翻转)
            c = int(np.clip(next_x, 0, N - 1))
            r = int(np.clip(M - 1 - next_y, 0, M - 1))

            if obstacles_map[r, c] == -1:
                # 撞到障碍物：位置回退到上一步，并掉头
                next_x = self.pos[0]
                next_y = self.pos[1]
                self.phi += np.pi  # 简单掉头处理

        # 确认更新
        self.pos = np.array([next_x, next_y])

    def calculate_bid(self, task):
        # 简单的拍卖计算逻辑
        task_pos = np.array(task['coords'])
        priority = task['priority']
        distance = np.linalg.norm(self.pos - task_pos)
        time_cost = distance / self.v
        bid = 0.5 * priority / 5 - 0.5 * (time_cost / 1200.0)  # 归一化处理
        return bid

    def calculate_reward(self, prev_entropy, curr_entropy, is_detected, action, map_size, obstacles_map, all_uavs):
        """
        最小化熵 + 探测保持 + 安全约束
        """
        # 1. 信息增益 (熵减)
        # 放大系数 10.0 是经验值（调节RL收敛速度），不影响物理参数
        r_info = (prev_entropy - curr_entropy) * 10.0

        # 2. 探测奖励 (Detection)
        # 探测到目标给予高额奖励
        r_detect = 10.0 if is_detected else 0.0

        # 3. 动作与安全 (保持你原有的参数不变)
        r_action = -0.05 * np.sum(action ** 2)

        r_collision = 0.0
        x, y = self.pos
        M, N = map_size

        # 边界约束 (你原来的参数)
        margin = 20.0
        if x < margin:
            r_collision -= (margin - x) / margin * 2.0
        elif x > N - margin:
            r_collision -= (x - (N - margin)) / margin * 2.0
        if y < margin:
            r_collision -= (margin - y) / margin * 2.0
        elif y > M - margin:
            r_collision -= (y - (M - margin)) / margin * 2.0
        if x < 0 or x >= N or y < 0 or y >= M:
            r_collision -= 10.0

        # 障碍物约束
        if obstacles_map is not None:
            c, r = int(np.clip(x, 0, N - 1)), int(np.clip(M - 1 - y, 0, M - 1))
            if obstacles_map[r, c] == -1:
                r_collision -= 10.0

        # 机间避碰 (你原来的参数 safe_dist=15.0)
        safe_dist = 15.0
        for other in all_uavs:
            if other is self:
                continue
            d = np.linalg.norm(self.pos - other.pos)
            if d < safe_dist:
                r_collision -= 5.0 * (1.0 - d / safe_dist)

        total_reward = r_info + r_detect + r_action + r_collision
        return total_reward


# 奖励归一化类部分
class RunningMeanStd:
    # 动态计算均值和方差的辅助类
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count


class RewardScaler:
    def __init__(self, shape, gamma=0.99, epsilon=1e-8):
        self.shape = shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x):
        # x: 单个奖励值或奖励数组
        # 更新统计量
        self.running_ms.update(x)
        # 归一化：这里只除以标准差 (Scaling)，不减均值 (Centering)
        # 这样可以保留奖励的正负符号（例如撞墙仍然是负的），只缩放幅度
        x_norm = x / np.sqrt(self.running_ms.var + self.epsilon)
        return x_norm
