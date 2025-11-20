# run_llm_online.py
import os

import numpy as np

from session_core import ConversationSession


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先在环境变量中设置 OPENAI_API_KEY。")
        return

    session = ConversationSession()

    print("==== User x LLM x LatentAligner 在线对齐 Demo ====")
    print("说明：")
    print("  - 你可以当我是一个会“不断学你的偏好”的 AI 助手。")
    print("  - 每一轮：你自然说话 → 系统根据 latent 偏好估计来回复 →")
    print("            下一句你的自然反应会被 LLM 当作“满意度线索”，")
    print("            我们把它转成模糊 reward，喂给对齐器偷偷调风格。")
    print("  - 对齐器会在隐空间里逐渐调整、甚至升维，不再打扰你打分。")
    print("输入 q / quit 结束。\n")

    turn = 0
    while True:
        print("\n" + "=" * 60)
        user_msg = input(f"[User] 第 {turn} 轮，你说：\n> ").strip()
        if user_msg.lower() in ["q", "quit", "exit"]:
            print("结束对话。")
            break

        result = session.handle_message(user_msg)

        reply = result["assistant_reply"]
        print("\n[AI] 回复：")
        print(reply)

        debug = result.get("debug", {})
        if debug:
            if "reward" in debug:
                print(
                    f"\n[DEBUG] 上一轮 reward ≈ {debug['reward']:+.3f}，"
                    f"预测 ≈ {debug.get('prediction', 0):+.3f}，"
                    f"对齐信号 e ≈ {debug.get('error', 0):+.3f}"
                )
            if "dim_update" in debug:
                dim = debug["dim_update"]
                if dim.get("expanded") is False:
                    print(
                        "  >>> 触发升维条件，但根据残差信号暂未扩展 "
                        f"(mean reward ≈ {dim['mean_reward']:+.3f}, ||grad|| ≈ {dim['residual_norm']:.3f})"
                    )
                else:
                    print(
                        "  >>> reward 长期偏低 + 残差强，升维 "
                        f"{dim['previous_k']} → {dim['new_k']} "
                        f"(mean reward ≈ {dim['mean_reward']:+.3f}, ||grad|| ≈ {dim['residual_norm']:.3f})"
                    )

        token_stats = result.get("stats", {}).get("token_stats", {})
        last_reply_tokens = (
            token_stats.get("last", {})
            .get("reply", {})
            .get("total_tokens")
        )
        if last_reply_tokens is not None:
            print(f"[DEBUG] 本轮回复 token 消耗 ≈ {last_reply_tokens}")

        turn += 1

    # 收尾：给你一点对齐器内部的“对用户的看法”
    print("\n====== 对齐器内部总结（数学视角） ======")
    stats = session.stats()
    print(f"最终子空间维度 k = {stats['current_k']}")
    if stats["dim_events"]:
        print("升维事件：")
        for event in stats["dim_events"]:
            print(
                "  - 第 {step} 轮: k {old} → {new} (mean reward ≈ {mr:+.3f}, ||grad|| ≈ {grad:.3f})".format(
                    step=event.get("step"),
                    old=event.get("previous_k"),
                    new=event.get("new_k"),
                    mr=event.get("mean_reward", 0.0),
                    grad=event.get("residual_norm", 0.0),
                )
            )
    else:
        print("整个对话过程中没有触发升维（要么轮数少，要么 reward 比较可预测）。")

    w_hat = session.aligner.current_approx_pref()
    print("\n[DEBUG] 最终 latent 偏好近似向量 w_hat（前 10 维）：")
    print(np.round(w_hat[:10], 3))
    print("（这里只是给技术同事看的，可选展示）")


if __name__ == "__main__":
    main()
