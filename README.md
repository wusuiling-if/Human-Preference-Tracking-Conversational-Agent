# 🧠 Human-Preference-Tracking Conversational Agent

**[ 动态偏好追踪 · 潜变量对齐 · 拟人化记忆 ]**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)

---

### 💡 告别“健忘”的 LLM，打造真正懂你的数字生命。

**HPT-Agent** 是一个前沿的智能体框架，旨在解决大模型在长程交互中的个性化缺失问题。它通过**实时偏好建模**与**潜变量映射（Latent Alignment）**，让 AI 能够随着对话深入，自动“进化”出最适合用户的沟通策略。

[查看演示 Demo] • [阅读文档] • [报告 Bug]

</div>

---

## ✨ 核心特性 (Why this project?)

| 🔥 实时偏好追踪 | 🎛️ 潜变量对齐 (Latent Alignment) |
| :--- | :--- |
| 拒绝静态画像。系统基于**语义流**实时计算 User Embedding，捕捉用户在情绪、话题、逻辑深度上的微小变化。 | 独创的**参数映射层**。将抽象的“偏好向量”数学化地映射为 LLM 的控制参数（如 Temperature, Presence Penalty, Tone 指令）。 |

| 🧬 拟人化仿真环境 | 🧠 闭环记忆系统 |
| :--- | :--- |
| 内置 `UserEnv`，支持生成具有 **Big Five (大五人格)** 特征的虚拟用户，用于低成本的大规模强化学习 (RL) 训练。 | 并不是简单的 Vector DB 检索，而是基于**Session Core** 的状态机管理，实现对话策略的动态切换。 |

---

## 🛠️ 系统架构 (Architecture)

本系统采用双循环架构：**外层对话循环**处理交互，**内层认知循环**处理偏好更新与对齐。

```mermaid
graph TD
    subgraph "User Environment / Real World"
        U[👤 用户 User] <-->|Interaction| FE[Web 前端 / API]
    end

    subgraph "Agent Core (Brain)"
        FE -->|Input Text| SC[⚙️ Session Core]
        
        %% 偏好分析链路
        SC -->|Analyze Stream| PT[🔍 偏好追踪器 Preference Tracker]
        PT -->|Update| PV[("🧬 偏好向量 (Embedding)"))]
        
        %% 对齐链路
        PV -->|Vector State| LA[🎛️ Latent Aligner]
        LA -->|Hyper-params & SysPrompt| LB[🌉 LLM Bridge]
        
        %% 生成链路
        LB <-->|Inference| LLM[🤖 大语言模型 (GPT/Local)]
        LLM -->|Response| SC
    end

    %% 可视化
    PV -.->|Real-time Data| VIZ[📊 Web 可视化面板]
⚡ 快速上手 (Quick Start)
1. 环境准备
Bash

git clone [https://github.com/wusuiling-if/Human-Preference-Tracking-Conversational-Agent.git](https://github.com/wusuiling-if/Human-Preference-Tracking-Conversational-Agent.git)
cd Human-Preference-Tracking-Conversational-Agent

# 推荐使用 Conda 或 venv
pip install -r requirements.txt
2. 配置密钥
Bash

# Linux / Mac
export OPENAI_API_KEY="sk-xxxx..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-xxxx..."
3. 启动全栈演示 (Web Dashboard)
启动后，你将看到一个实时变化的偏好雷达图，展示 AI 如何理解你的兴趣。

Bash

python web_server.py
# 访问 http://localhost:8000
🔬 核心代码预览
Latent Aligner 是本项目的灵魂。它如何将抽象的“用户喜欢严谨”转化为代码逻辑？

Python

# latent_aligner.py (Simplified)

def align_model_parameters(preference_vector):
    """
    将偏好向量动态映射为 LLM 生成参数
    """
    params = {
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # 维度 0: 创造性 vs 严谨性
    if preference_vector[0] > 0.5:
        # 用户喜欢发散思维 -> 提高温度
        params["temperature"] = 0.7 + (preference_vector[0] * 0.5)
    else:
        # 用户喜欢严谨事实 -> 降低温度，增加惩罚
        params["temperature"] = 0.3
        params["frequency_penalty"] = 0.5
        
    # 维度 1: 简洁 vs 详尽
    if preference_vector[1] > 0.8:
        params["max_tokens"] = 150  # 强制简短
        
    return params
📅 路线图 (Roadmap)
[x] Phase 1: 基础架构搭建，实现实时偏好向量更新。

[x] Phase 2: Web 可视化前端，支持 WebSocket 实时数据流。

[ ] Phase 3: 引入 RLHF (Reinforcement Learning from Human Feedback) 接口，让 Agent 自我博弈。

[ ] Phase 4: 支持本地量化模型 (Llama 3 / Mistral) 的端侧部署。

[ ] Phase 5: 长期记忆向量库 (Vector DB) 集成。

🤝 参与贡献
我们非常欢迎 Pull Requests！ 如果你对 计算心理学、人机交互 (HCI) 或 LLM 微调 感兴趣，请加入我们。

Fork 本仓库

创建分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送 (git push origin feature/AmazingFeature)

提交 PR
