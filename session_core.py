"""Shared conversation session logic for CLI and web interfaces."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    D_REAL,
    INIT_K,
    MAX_K,
    LAMBDA_RIDGE,
    WINDOW,
    SEED,
    EXPLORE_PROB,
    RESIDUAL_NORM_THRESH,
    BAD_MEAN_THRESH,
)
from latent_aligner import LatentAligner
from llm_bridge import LLMBridge


class ConversationSession:
    """Stateful wrapper around LatentAligner + LLMBridge."""

    def __init__(self) -> None:
        rng = np.random.default_rng(SEED)
        self.aligner = LatentAligner(
            D=D_REAL,
            k_init=INIT_K,
            k_max=MAX_K,
            lam=LAMBDA_RIDGE,
            rng=rng,
            explore_prob=EXPLORE_PROB,
        )
        self.bridge = LLMBridge()
        self.conversation: List[Tuple[str, str]] = []
        self.recent_errors: List[float] = []
        self.reward_history: List[float] = []
        self.dim_events: List[Dict[str, Any]] = []
        self.pending_action: Optional[np.ndarray] = None
        self.turn = 0

        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"session_{ts_label}.log"
        self._prune_logs(log_dir, keep=10)
        self.total_tokens = {
            "reply": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "reward": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self.last_tokens = {
            "reply": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "reward": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self.style_hint = ""

    # ----------------------------- helpers ---------------------------------
    def stats(self) -> Dict:
        w_hat = self.aligner.current_approx_pref()
        return {
            "turn": self.turn,
            "current_k": self.aligner.k,
            "reward_history": self.reward_history[-100:],
            "recent_mse": float(np.mean(np.square(self.recent_errors[-WINDOW:])))
            if len(self.recent_errors) >= WINDOW
            else None,
            "dim_events": [event.copy() for event in self.dim_events[-20:]],
            "w_hat_preview": list(np.round(w_hat[:8], 3)),
            "token_stats": {
                "total": {k: v.copy() for k, v in self.total_tokens.items()},
                "last": {k: v.copy() for k, v in self.last_tokens.items()},
            },
            "style_hint": self.style_hint,
        }

    def conversation_tail(self, limit: int = 20) -> List[Dict[str, str]]:
        return [
            {"role": role, "content": content}
            for role, content in self.conversation[-limit:]
        ]

    def _accumulate_tokens(self, key: str, usage: Dict[str, int]) -> None:
        target_total = self.total_tokens.get(key)
        target_last = self.last_tokens.get(key)
        if target_total is None or target_last is None:
            return
        for token_type in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = int(usage.get(token_type, 0))
            target_total[token_type] += value
            target_last[token_type] = value

    def _write_log(self, payload: Dict) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            **payload,
        }
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _prune_logs(log_dir: Path, keep: int) -> None:
        logs = sorted(log_dir.glob("session_*.log"))
        if len(logs) <= keep:
            return
        for stale in logs[:-keep]:
            try:
                stale.unlink()
            except OSError:
                pass

    def _maybe_expand(self, turn_index: int) -> Optional[Dict[str, object]]:
        """Expand子空间仅在 reward 长期偏低且残差信号强时触发。"""
        if self.aligner.k >= MAX_K:
            return None

        residual_norm = float(np.linalg.norm(self.aligner.grad_residual))

        if len(self.reward_history) < WINDOW:
            return None

        recent_rewards = self.reward_history[-WINDOW:]
        mean_reward = float(np.mean(recent_rewards))

        should_expand = (
            mean_reward < BAD_MEAN_THRESH
            and residual_norm > RESIDUAL_NORM_THRESH
            and (turn_index % WINDOW == 0)
        )

        reason: Dict[str, Any] = {}
        if not should_expand:
            return None

        before_k = self.aligner.k
        expanded = self.aligner.expand_subspace(min_norm=RESIDUAL_NORM_THRESH)
        reason.setdefault("mean_reward", mean_reward)
        reason.setdefault("residual_norm", residual_norm)
        reason["previous_k"] = before_k
        reason["new_k"] = self.aligner.k

        if expanded:
            self.dim_events.append(
                {
                    "step": turn_index,
                    "previous_k": before_k,
                    "new_k": self.aligner.k,
                    "mean_reward": mean_reward,
                    "residual_norm": residual_norm,
                }
            )
            reason["expanded"] = True
        else:
            reason["expanded"] = False
        return reason

    # ----------------------------- main API --------------------------------
    def handle_message(self, user_msg: str) -> Dict:
        debug_info: Dict = {}

        # 1) 如果有上一轮的 action，用本次自然输入估计 reward
        if self.pending_action is not None:
            reward, reward_usage, hard_flags = self.bridge.estimate_reward(
                self.conversation, user_msg
            )
            soft_reward = reward
            if "forbid_parentheses" in hard_flags:
                soft_reward = 0.0

            e, r_hat = self.aligner.update_with_sample(self.pending_action, soft_reward)
            # recent_errors 现在记录 reward/advantage 信号，而非预测误差
            self.recent_errors.append(e)
            self.reward_history.append(reward)
            debug_info.update(
                {
                    "reward": reward,
                    "soft_reward": soft_reward,
                    "prediction": r_hat,
                    "error": e,
                    "hard_flags": hard_flags,
                }
            )

            self._accumulate_tokens("reward", reward_usage)

            if reward < 0:
                self.style_hint = f"上一轮用户不满，抱怨内容：{user_msg[:200]}"
            elif reward > 0.2:
                self.style_hint = f"上一轮用户喜欢这种语气：{user_msg[:200]}"
            else:
                self.style_hint = ""

            expand_info = self._maybe_expand(self.turn)
            if expand_info:
                debug_info["dim_update"] = expand_info

        # 2) 当前输入触发新的回复
        action_vec = self.aligner.sample_action()
        reply, reply_usage = self.bridge.generate_reply(
            action_vec,
            self.conversation,
            user_msg,
            style_hint=self.style_hint,
        )
        self.conversation.append(("user", user_msg))
        self.conversation.append(("assistant", reply))
        self._accumulate_tokens("reply", reply_usage)

        self.pending_action = action_vec
        self.turn += 1

        self._write_log(
            {
                "turn": self.turn,
                "user": user_msg,
                "assistant": reply,
                "reward": debug_info.get("reward"),
                "prediction": debug_info.get("prediction"),
                "error": debug_info.get("error"),
                "k": self.aligner.k,
                "tokens": {
                    "last": {k: v.copy() for k, v in self.last_tokens.items()},
                    "total": {k: v.copy() for k, v in self.total_tokens.items()},
                },
                "style_hint": self.style_hint,
            }
        )

        return {
            "assistant_reply": reply,
            "debug": debug_info,
            "stats": self.stats(),
            "conversation": self.conversation_tail(),
        }

    def snapshot(self) -> Dict:
        return {
            "stats": self.stats(),
            "conversation": self.conversation_tail(),
        }
