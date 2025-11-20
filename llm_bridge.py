# llm_bridge.py
import os
import json
from typing import List, Tuple, Dict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 允许从 .env 中加载 API key / 模型配置
load_dotenv()

# 可以根据你账号的权限改成别的模型名，也可以通过环境变量覆盖
LLM_MODEL_ACTOR = os.getenv("LLM_MODEL_ACTOR", "deepseek-chat")
LLM_MODEL_REWARD = os.getenv("LLM_MODEL_REWARD", "deepseek-chat")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")


class LLMBridge:
    """
    LLM 与 latent 对齐器之间的桥接层：

    - generate_reply:
        给定 latent action 向量 + 对话历史 + 当前用户输入，
        让 LLM 按这个“行为向量”控制说话风格，生成回复文本。

    - estimate_reward_from_reaction:
        给定最近几轮对话 + 用户对上一轮回复后的自然反应，
        让 LLM 读出“上一轮风格在多大程度上让用户满意”，输出 reward ∈ [-1, 1]。
    """

    def __init__(self, model_actor: str = None, model_reward: str = None) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("请先在环境变量中设置 DEEPSEEK_API_KEY（或 OPENAI_API_KEY）。")

        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_API_BASE.rstrip("/"))
        self.model_actor = model_actor or LLM_MODEL_ACTOR
        self.model_reward = model_reward or LLM_MODEL_REWARD

    # ---------------------------------------------------------------------
    # 1) latent action → LLM 回复（风格控制）
    # ---------------------------------------------------------------------
    def _format_action_profile(self, action_vec: np.ndarray) -> str:
        v = np.asarray(action_vec, dtype=float)
        if v.ndim != 1:
            v = v.ravel()
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        else:
            v = np.zeros_like(v)

        style_code = ",".join(f"{val:+.3f}" for val in v)
        profile = (
            "你是一名可塑的 AI 伙伴。系统会提供一个 style_code（一个实数序列），"
            "它并没有固定含义，但要求你满足：相似的 code → 相似的风格/语调/节奏。\n"
            "接下来请：\n"
            "1. 把 style_code 当作隐式标签，自行决定最合适的表达方式；\n"
            "2. 用自然、人类化的语言回复用户，不要提及 code 的具体值或含义；\n"
            "3. 记住：唯一的目标是让用户觉得好交流、好理解。\n"
            f"本轮 style_code: [{style_code}]\n"
        )
        return profile

    def generate_reply(
        self,
        action_vec: np.ndarray,
        conversation: List[Tuple[str, str]],
        user_msg: str,
        style_hint: str = "",
    ) -> Tuple[str, Dict[str, int]]:
        """
        使用当前的 latent action 向量，控制 LLM 回复风格。
        conversation: [(role, content), ...]，role ∈ {"user", "assistant"}
        """
        sys_prompt = self._format_action_profile(action_vec)
        if style_hint:
            sys_prompt += "\n\n用户最近的情绪/偏好提示：" + style_hint

        messages = [{"role": "system", "content": sys_prompt}]

        # 带一点历史（最近几轮）
        for role, content in conversation[-6:]:
            api_role = "user" if role == "user" else "assistant"
            messages.append({"role": api_role, "content": content})

        # 当前这轮的用户输入
        messages.append({"role": "user", "content": user_msg})

        resp = self.client.chat.completions.create(
            model=self.model_actor,
            messages=messages,
            temperature=0.7,
        )
        usage = resp.usage or None
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        return resp.choices[0].message.content.strip(), usage_dict

    # ---------------------------------------------------------------------
    # 2) 用户自然反应 → 对上一轮的 reward
    # ---------------------------------------------------------------------
    def estimate_reward(
        self,
        conversation: List[Tuple[str, str]],
        user_reaction_text: str,
    ) -> Tuple[float, Dict[str, int], List[str]]:
        """
        根据“用户在上一轮 AI 回复之后的那句自然反应”估计上一轮的 reward。

        - 不要求用户显式评价风格；
        - LLM 自己从语气/内容/情绪里读出“爽不爽”；
        - 返回一个标量 reward ∈ [-1, 1]。
        """
        # 准备最近几轮对话文本
        history_text = ""
        for role, content in conversation[-6:]:
            tag = "用户" if role == "user" else "AI"
            history_text += f"[{tag}]: {content}\n"

        sys_prompt = (
            "你是一个“对话风格满意度评估器”。\n"
            "现在要判断：用户对 AI **上一轮的回复**，在“说话方式/风格/语气”上有多满意。\n\n"
            "你会看到：\n"
            "1. 最近几轮完整对话；\n"
            "2. 用户在上一轮 AI 回复之后，给出的下一句自然反应\n"
            "   （可能是继续深入、略过、抱怨、质疑、转移话题等等）。\n\n"
            "请你：\n"
            "  - 只从“风格/语气/结构”角度判断，不考虑答案事实对不对；\n"
            "  - 如果用户明显认可、愿意继续深入、显得轻松配合 → reward 靠近 1；\n"
            "  - 如果用户冷淡、觉得被误解、明显不爽或想赶紧结束 → reward 靠近 -1；\n"
            "  - 中性、没什么态度 → reward 接近 0；\n"
            "  - 输出一个 JSON：{\"reward\": 数值, \"hard_flags\": [..]}，reward 范围 [-1,1]；\n"
            "    - hard_flags 是可选列表，例如当你发现用户明确要求“不要括号”时，可输出 [\"forbid_parentheses\"]；\n"
            "  - 不要输出多余文字，只输出 JSON。\n"
        )

        payload = {
            "recent_conversation": history_text,
            "user_reaction_after_last_ai_reply": user_reaction_text,
        }

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": (
                    "下面是信息（JSON 格式）：\n"
                    + json.dumps(payload, ensure_ascii=False)
                    + "\n\n请直接给出 reward JSON。"
                ),
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model_reward,
            messages=messages,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            r = float(parsed.get("reward", 0.0))
            hard_flags = parsed.get("hard_flags", []) or []
        except Exception:
            r = 0.0
            hard_flags = []

        # 裁剪到 [-1, 1]
        r = max(-1.0, min(1.0, r))
        usage = resp.usage or None
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
        return r, usage_dict, [str(flag) for flag in hard_flags]

    # 为兼容老代码，保留旧 API 名称
    def estimate_reward_from_reaction(
        self,
        conversation: List[Tuple[str, str]],
        user_reaction_text: str,
    ) -> Tuple[float, Dict[str, int], List[str]]:
        return self.estimate_reward(conversation, user_reaction_text)
