import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import maddpg_v2 as RL
import targetpre_v2 as tp

os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def moving_average(data, window):
    if len(data) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(np.array(data), kernel, mode="valid")


def save_reward_history(output_dir, reward_history, noise_history):
    ensure_dir(output_dir)

    csv_path = os.path.join(output_dir, "reward_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("episode,avg_reward,noise_std\n")
        for idx, reward in enumerate(reward_history, start=1):
            noise = noise_history[idx - 1] if idx - 1 < len(noise_history) else None
            f.write(f"{idx},{reward:.6f},{noise:.6f}\n")

    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, label="Avg Reward")
    ma_50 = moving_average(reward_history, 50)
    ma_100 = moving_average(reward_history, 100)
    if ma_50.size > 0:
        plt.plot(range(49, 49 + len(ma_50)), ma_50, label="MA50")
    if ma_100.size > 0:
        plt.plot(range(99, 99 + len(ma_100)), ma_100, label="MA100")
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()


def save_config(output_dir, config):
    ensure_dir(output_dir)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def init_predictors_targets(map_size, obstacles, tarcfgs):
    predictors = []
    real_targets = []
    for cfg in tarcfgs:
        start_belief = cfg["pos"]
        v_range = cfg["v_range"]
        theta_range = cfg["theta_range"]
        ID = cfg["ID"]
        priority = cfg["priority"]
        p = tp.TargetPredictor(map_size, obstacles, v_range, theta_range, start_belief, ID=ID, priority=priority)

        initial_v = cfg["initial_v"]
        initial_phi = cfg["initial_phi"]
        r = tp.RealTarget(ID, priority, start_belief, initial_v, initial_phi)

        predictors.append(p)
        real_targets.append(r)
    return predictors, real_targets


def compute_local_entropy_for_uavs(uav_list, predictors):
    local_entropies = []
    for uav in uav_list:
        local_entropy = 0.0
        for predictor in predictors:
            local_entropy += predictor.get_local_entropy(uav.pos, uav.detect_radius)
        local_entropies.append(local_entropy)
    return local_entropies


def plot_trajectories(output_dir, map_size, obs_map, uav_trajectories, target_trajectories, detection_points, eval_tag):
    ensure_dir(output_dir)
    plt.figure(figsize=(8, 8))
    if obs_map is not None:
        plt.imshow(obs_map == -1, cmap="gray_r", origin="upper")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, traj in enumerate(uav_trajectories):
        traj = np.array(traj)
        if traj.size == 0:
            continue
        color = colors[idx % len(colors)]
        plt.plot(traj[:, 0], traj[:, 1], color=color, label=f"UAV {idx + 1}")

    for idx, traj in enumerate(target_trajectories):
        traj = np.array(traj)
        if traj.size == 0:
            continue
        plt.plot(traj[:, 0], traj[:, 1], linestyle="--", linewidth=2.5, label=f"Target {idx + 1}")

    if detection_points:
        det_points = np.array(detection_points)
        plt.scatter(det_points[:, 0], det_points[:, 1], s=20, c="red", label="Detection")

    plt.xlim(0, map_size[1])
    plt.ylim(0, map_size[0])
    plt.title("UAV Trajectories & Detections")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trajectory_{eval_tag}.png"))
    plt.close()


def plot_entropy_curve(output_dir, entropy_curve, eval_tag):
    ensure_dir(output_dir)
    plt.figure(figsize=(8, 4))
    plt.plot(entropy_curve, label="Entropy")
    plt.title("Entropy Curve")
    plt.xlabel("Time")
    plt.ylabel("H(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"entropy_curve_{eval_tag}.png"))
    plt.close()


def plot_heatmaps(output_dir, predictors, time_step, eval_tag):
    for predictor in predictors:
        grid = predictor._generate_grid()
        plt.figure(figsize=(6, 5))
        plt.imshow(grid, cmap="hot", origin="upper")
        plt.colorbar(label="Belief")
        plt.title(f"Target {predictor.ID} Heatmap t={time_step}")
        plt.tight_layout()
        filename = f"heatmap_target{predictor.ID}_t{time_step}_{eval_tag}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


def run_evaluation(eval_dir, episode, uav_list, uav_configs, map_size, obs_map, obstacles, tarcfgs, max_steps):
    predictors, real_targets = init_predictors_targets(map_size, obstacles, tarcfgs)
    for i, uav in enumerate(uav_list):
        uav.pos = np.array(uav_configs[i]["pos"], dtype=float)
        uav.v = uav_configs[i]["initial_v"]
        uav.phi = np.radians(uav_configs[i]["phi"])

    uav_trajectories = [[] for _ in uav_list]
    target_trajectories = [[] for _ in real_targets]
    detection_points = []
    entropy_curve = []
    time_points = {0, 25, 50, 75, 99}

    for uav in uav_list:
        uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

    for t in range(max_steps):
        if t in time_points:
            plot_heatmaps(eval_dir, predictors, t, f"ep{episode}")

        entropy_curve.append(sum(p.get_entropy() for p in predictors))

        action_list = []
        for uav in uav_list:
            action = uav.brain.select_action(uav.last_obs)
            action_list.append(np.clip(action, -1.0, 1.0))

        for i, uav in enumerate(uav_list):
            uav.state_update(action_list[i], map_size, obs_map)
            uav_trajectories[i].append(uav.pos.copy())

        meas_noise_std = 3.0
        innovation_norm = np.zeros(len(predictors))
        for i, p in enumerate(predictors):
            real_pos = real_targets[i].state[0:2]
            target_trajectories[i].append(real_pos.copy())

            det_res = []
            sum_z = 0.0
            sum_detected = 0
            for u_idx, u in enumerate(uav_list):
                dist = np.linalg.norm(u.pos - real_pos)
                is_detected = (dist < 250.0) and (np.random.rand() < 0.9)
                temp_state = {"detected": is_detected, "measurement": None, "uavpos": u.pos, "uavdp": u.detecct_p}
                if is_detected:
                    detection_points.append(u.pos.copy())
                    temp_state["measurement"] = real_pos + np.random.randn(2) * meas_noise_std
                    sum_z += temp_state["measurement"]
                    sum_detected += 1
                det_res.append(temp_state)

            if sum_detected == 0:
                z_average = None
                p.step_update(None, innovation_norm[i], det_res)
                innovation_norm[i] = 0.0
            else:
                z_average = sum_z / sum_detected
                p.step_update(z_average, innovation_norm[i], det_res)
                innovation_norm[i] = np.linalg.norm(z_average - p.state_si[0:2])

            real_targets[i].step_forward()

        for uav in uav_list:
            uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

    eval_tag = f"ep{episode}"
    plot_trajectories(eval_dir, map_size, obs_map, uav_trajectories, target_trajectories, detection_points, eval_tag)
    plot_entropy_curve(eval_dir, entropy_curve, eval_tag)

    eval_h_final = entropy_curve[-1] if entropy_curve else 0.0
    eval_delta_h = (entropy_curve[0] - entropy_curve[-1]) if len(entropy_curve) > 1 else 0.0
    return eval_h_final, eval_delta_h


# 主要训练逻辑
def train_with_improvements():
    # 1. 初始化环境与参数
    map_size = (2000, 2000)
    # 简单的障碍物 (Row, Col)
    obstacles = [(600, 600), (600, 601), (601, 600), (601, 601),
                 (1200, 1200), (1200, 1201), (1201, 1200), (1201, 1201)]

    # 转换障碍物地图
    obs_map = np.zeros(map_size)
    for r, c in obstacles:
        if 0 <= r < map_size[0] and 0 <= c < map_size[1]:
            obs_map[r, c] = -1

    # 定义目标配置
    tarcfgs = [
        {"ID": 1, "pos": (500.0, 500.0), "priority": 1, "v_range": (15, 60),
         "theta_range": (np.radians(15), np.radians(60)), "initial_v": 20, "initial_phi": 45},
        {"ID": 2, "pos": (1500.0, 1500.0), "priority": 1, "v_range": (15, 60),
         "theta_range": (np.radians(225), np.radians(270)), "initial_v": 20, "initial_phi": -120}
    ]

    # 初始化 MADDPG 系统
    state_dim = 17
    action_dim = 2
    # 无人机参数配置
    uav_configs = [
        {"id": 1, "pos": [200, 200], "phi": 45, "initial_v": 30},
        {"id": 2, "pos": [1800, 1800], "phi": 225, "initial_v": 30},
        {"id": 3, "pos": [200, 1800], "phi": 315, "initial_v": 30},
        {"id": 4, "pos": [1800, 200], "phi": 135, "initial_v": 30},
        {"id": 5, "pos": [1000, 1000], "phi": 0, "initial_v": 0}
    ]

    num_uavs = len(uav_configs)
    total_state_dim = state_dim * num_uavs
    total_action_dim = action_dim * num_uavs

    uav_list = []
    for cfg in uav_configs:
        uav = RL.UAVAgent(
            uav_id=cfg["id"],
            initial_pos=cfg["pos"],
            initial_v=cfg["initial_v"],
            initial_phi=cfg["phi"],
            step=1.0,
            total_state_dim=total_state_dim,
            total_action_dim=total_action_dim
        )

        uav.state_dim = state_dim
        uav.brain = RL.MADDPG(state_dim, action_dim, uav.max_action_tensor, total_state_dim, total_action_dim)

        uav_list.append(uav)

    # 初始化全局 Buffer
    global_buffer = RL.MultiAgentReplayBuffer(50000, num_uavs,
                                              [state_dim] * num_uavs,
                                              [action_dim] * num_uavs)

    reward_history = []
    noise_history = []
    MAX_EPISODES = 15000
    MAX_STEPS = 100
    BATCH_SIZE = 512
    noise_std = 0.3
    min_noise = 0.05
    noise_decay = 0.996
    eval_interval = 200

    reward_scaler = RL.RewardScaler(shape=(1,))  # 初始化奖励归一化器

    output_root = os.path.join(os.path.dirname(__file__), "outputs")
    models_dir = os.path.join(output_root, "models")
    eval_dir = os.path.join(output_root, "eval")
    ensure_dir(models_dir)
    ensure_dir(eval_dir)

    config = {
        "MAP_SIZE": map_size,
        "N": num_uavs,
        "MAX_STEPS": MAX_STEPS,
        "MAX_EPISODES": MAX_EPISODES,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "gamma": 0.99,
        "seed": None,
        "eval_interval": eval_interval
    }
    save_config(output_root, config)

    best_record_path = os.path.join(output_root, "best_record.json")
    best_record = {"episode": -1, "eval_H_final": float("inf"), "eval_delta_H": -float("inf")}
    if os.path.exists(best_record_path):
        with open(best_record_path, "r", encoding="utf-8") as f:
            best_record = json.load(f)

    # 训练主循环
    print("开始训练 Improved MADDPG")
    for episode in range(MAX_EPISODES):
        predictors, real_targets = init_predictors_targets(map_size, obstacles, tarcfgs)

        # 重置无人机部分参数
        for i, uav in enumerate(uav_list):
            uav.pos = np.array(uav_configs[i]["pos"])
            uav.v = uav_configs[i]["initial_v"]
            uav.phi = np.radians(uav_configs[i]["phi"])
            uav.assigned_task_coords = None

        episode_reward = 0

        # 预先获取初始观测
        for uav in uav_list:
            uav.last_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)

        for t in range(MAX_STEPS):
            local_entropy_before = compute_local_entropy_for_uavs(uav_list, predictors)

            action_list = []
            obs_list = []

            for uav in uav_list:
                obs = uav.last_obs
                obs_list.append(obs)

                raw_action = uav.brain.select_action(obs)
                noise = np.random.normal(0, noise_std, size=2)
                action = np.clip(raw_action + noise, -1.0, 1.0)
                action_list.append(action)

            # 执行动作 & 环境更新
            next_obs_list = []
            reward_list = []
            done_list = []
            uav_detection_states = [False] * len(uav_list)

            # 无人机运动
            for i, uav in enumerate(uav_list):
                uav.state_update(action_list[i], map_size, obs_map)

            # 预测器更新
            meas_noise_std = 3.0
            innovation_norm = np.zeros(len(predictors))
            for i, p in enumerate(predictors):
                real_pos = real_targets[i].state[0:2]

                det_res = []
                sum_z = 0.0
                sum_detected = 0
                for u_idx, u in enumerate(uav_list):
                    dist = np.linalg.norm(u.pos - real_pos)
                    is_detected = (dist < 250.0) and (np.random.rand() < 0.9)
                    temp_state = {"detected": is_detected, "measurement": None, "uavpos": u.pos, "uavdp": u.detecct_p}
                    if is_detected:
                        uav_detection_states[u_idx] = True
                        temp_state["measurement"] = real_pos + np.random.randn(2) * meas_noise_std
                        sum_z += temp_state["measurement"]
                        sum_detected += 1
                    det_res.append(temp_state)

                if sum_detected == 0:
                    z_average = None
                    p.step_update(None, innovation_norm[i], det_res)
                    innovation_norm[i] = 0.0
                else:
                    z_average = sum_z / sum_detected
                    p.step_update(z_average, innovation_norm[i], det_res)
                    innovation_norm[i] = np.linalg.norm(z_average - p.state_si[0:2])

                # 真实目标移动
                real_targets[i].step_forward()

            local_entropy_after = compute_local_entropy_for_uavs(uav_list, predictors)

            # 观测下一帧 & 计算奖励
            for i, uav in enumerate(uav_list):
                next_obs = uav.get_observation(map_size, obs_map, uav_list, predictors)
                next_obs_list.append(next_obs)
                uav.last_obs = next_obs

                r = uav.calculate_reward(
                    prev_entropy=local_entropy_before[i],
                    curr_entropy=local_entropy_after[i],
                    is_detected=uav_detection_states[i],
                    action=action_list[i],
                    map_size=map_size,
                    obstacles_map=obs_map,
                    all_uavs=uav_list
                )

                r_input = np.array([r])
                r_norm = reward_scaler(r_input)[0]
                r_final = np.clip(r_norm, -5.0, 5.0)

                reward_list.append(r_final)

                d = False
                if t == MAX_STEPS - 1:
                    d = True
                done_list.append(d)

                episode_reward += r_final

            global_buffer.add(obs_list, action_list, reward_list, next_obs_list, done_list)

            if t % 5 == 0 and global_buffer.size > BATCH_SIZE:
                for _ in range(2):
                    RL.train_centralized(uav_list, global_buffer, BATCH_SIZE)

        noise_std = max(min_noise, noise_std * noise_decay)
        avg_reward = episode_reward / len(uav_list)
        reward_history.append(avg_reward)
        noise_history.append(noise_std)
        print(f"Episode {episode + 1}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Noise: {noise_std:.3f}")

        if (episode + 1) % 50 == 0:
            for uav in uav_list:
                torch.save(uav.brain.actor.state_dict(), os.path.join(models_dir, f"uav{uav.ID}_actor_{episode + 1}.pth"))

        if (episode + 1) % eval_interval == 0:
            eval_h_final, eval_delta_h = run_evaluation(
                eval_dir, episode + 1, uav_list, uav_configs, map_size, obs_map, obstacles, tarcfgs, MAX_STEPS
            )

            if eval_h_final < best_record["eval_H_final"] or eval_delta_h > best_record["eval_delta_H"]:
                for uav in uav_list:
                    torch.save(uav.brain.actor.state_dict(), os.path.join(models_dir, f"best_actor_uav{uav.ID}.pth"))

                best_record = {
                    "episode": episode + 1,
                    "eval_H_final": eval_h_final,
                    "eval_delta_H": eval_delta_h
                }
                with open(best_record_path, "w", encoding="utf-8") as f:
                    json.dump(best_record, f, indent=2, ensure_ascii=False)

    save_reward_history(output_root, reward_history, noise_history)


if __name__ == "__main__":
    train_with_improvements()
