# run_simulation.py
import numpy as np
from typing import List

from config import (
    D_REAL,
    INIT_K,
    MAX_K,
    NOISE_STD,
    LAMBDA_RIDGE,
    WINDOW,
    ERR_THRESH,
    T_STEPS,
    SEED,
    EXPLORE_PROB,
)
from user_env import UserEnv
from latent_aligner import LatentAligner


def main():
    rng = np.random.default_rng(SEED)

    # 1. 真实用户（我们不知道他的 w_true）
    user = UserEnv.random(dim=D_REAL, noise_std=NOISE_STD, rng=rng)

    # 2. 对齐器：一开始认为只有 INIT_K 维
    aligner = LatentAligner(
        D=D_REAL,
        k_init=INIT_K,
        k_max=MAX_K,
        lam=LAMBDA_RIDGE,
        rng=rng,
        explore_prob=EXPLORE_PROB,
    )

    recent_errors: List[float] = []
    dim_events = []

    print(f"真实用户偏好向量 w_true（前 8 维）：")
    print(np.round(user.true_pref()[:8], 3))
    print(f"\n初始子空间维度 k = {aligner.k}\n")

    for t in range(T_STEPS):
        # 系统选择一个行为向量（可以理解为某种说话风格 embedding）
        a = aligner.sample_action()

        # 用户给出模糊反馈
        r = user.step(a)

        # 系统根据当前子空间更新自身参数
        e, r_hat = aligner.update_with_sample(a, r)
        recent_errors.append(e)

        # 打一点点 log
        if (t + 1) % 50 == 0:
            mse_50 = float(np.mean(np.square(recent_errors[-50:])))
            print(
                f"step {t+1:4d} | k={aligner.k} | "
                f"r={r:+.3f} r_hat={r_hat:+.3f} | e={e:+.3f} | MSE_50={mse_50:.4f}"
            )

        # 每 WINDOW 步，检查要不要升维
        if (t + 1) % WINDOW == 0 and aligner.k < MAX_K:
            window_err = np.mean(np.square(recent_errors[-WINDOW:]))
            if window_err > ERR_THRESH:
                before_k = aligner.k
                expanded = aligner.expand_subspace()
                if expanded:
                    dim_events.append((t + 1, aligner.k))
                    print(
                        f"  >>> step {t+1}: 近期误差 {window_err:.4f} 偏大，"
                        f"从残差中挖出一条新方向，子空间升维 {before_k} -> {aligner.k}"
                    )

    # 最后看一下：最终逼近效果如何
    w_hat = aligner.current_approx_pref()
    cos_sim = float(
        np.dot(w_hat, user.true_pref())
        / (np.linalg.norm(w_hat) * np.linalg.norm(user.true_pref()) + 1e-9)
    )

    print("\n====== 结果小结 ======")
    print(f"最终子空间维度 k = {aligner.k}")
    if dim_events:
        print("升维事件：")
        for step, k in dim_events:
            print(f"  step {step:4d} -> k = {k}")
    else:
        print("无升维事件（初始子空间就够用，或阈值设置过宽）。")

    print("\n最终估计的用户偏好向量 w_hat（前 8 维）：")
    print(np.round(w_hat[:8], 3))
    print("\n真实 w_true（前 8 维）以作对比：")
    print(np.round(user.true_pref()[:8], 3))
    print(f"\ncosine 相似度 ≈ {cos_sim:.4f}")


if __name__ == "__main__":
    main()
