# Aider4IntraLLM

> 基于 [Aider](https://github.com/Aider-AI/aider) v0.86.1 进行二次开发版本

## 📋 项目背景

在公司内网环境下，目前主流的 AI 编程助手（如 ClaudeCode、CodeX、OpenCode等）由于需要登录验证、环境变量配置，无法在受限的内网环境中使用。为了满足内网开发团队的 **Vibe Coding** 需求，基于本人对aider的体验感受，对开源的 Aider 项目进行二次开发，打造一款适合在公司内网环境的 AI 编程助手。（注：公司需要在内网环境部署模型）

## 🔄 二开改造点

### 计划改造 📝


- [ ] 添加skills
    - 测试：tests/basic/test_skills.py
    - `python -m pytest -q tests/basic/test_skills.py`
- [ ] 新增agent模式，按如下步骤下编写代码
    - brainstorming - Socratic design refinement
    - writing-plans - Detailed implementation plans
    - executing-plans - Batch execution with checkpoints
    - dispatching-parallel-agents - Concurrent subagent workflows
    - requesting-code-review - Pre-review checklist
    - receiving-code-review - Responding to feedback
- [ ] 优化reflection机制
- [ ] 优化上下文工程
    - [hello_agents](https://github.com/datawhalechina/hello-agents/blob/main/docs/chapter9/%E7%AC%AC%E4%B9%9D%E7%AB%A0%20%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B.md)
    - [上下文工程skills](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering)
    - [Denis2054/Context-Engineering-for-Multi-Agent-Systems](https://github.com/Denis2054/Context-Engineering-for-Multi-Agent-Systems)
    - [LongCodeZip](https://github.com/YerbaPage/LongCodeZip)
- [ ] 添加网络搜索工具（只对内网访问白名单的网站进行网络搜索）
- [ ] 像 software-agent-sdk-main 那样：输入包含 trigger 关键词就自动注入对应 skill 内容”（无需手动 /skill load），我也可以继续帮你把自动触发注入加到 Aider 的消息构建流程里，并做开关配置，避免误触发。


## 项目本地启动

为了方便本地测试，可以注册一个`OpenRouter`账号

```bash
python -m venv venv
venv\Scripts\Activate
python -m pip install -U pip
pip install -r requirements.txt
$env:PYTHONWARNINGS="ignore"

# python -m aider --model openrouter/<provider>/<model>
$env:OPENROUTER_API_KEY="你的OpenRouter key"
python -m aider --model openrouter/deepseek/deepseek-v3.2
```


## 📚 使用文档

详细使用文档请参考：
- [原版 Aider 文档](./README_aider.md)

## 🔧 技术架构

```
aider_chat-Intranet/
├── aider/                      # 核心代码
│   ├── main.py                # 主入口
│   ├── coders/                # 编码器（多种编辑格式）
│   ├── models.py              # 模型管理
│   ├── repo.py                # Git 仓库管理
│   ├── repomap.py             # 代码库映射
│   ├── prompts.py             # 提示词
│   └── io.py                  # 交互界面
├── requirements.txt           # Python 依赖
├── pyproject.toml            # 项目配置
└── docs/                     # 文档目录
```


## 📄 许可证

本项目基于 Apache License 2.0 开源协议。

- 原项目：[Aider-AI/aider](https://github.com/Aider-AI/aider)
- 二次开发：遵循 Apache 2.0 协议，保留原作者版权声明

## 🙏 致谢

感谢 [Aider](https://github.com/Aider-AI/aider) 项目及其社区的贡献者们，为我们提供了如此优秀的开源 AI 编程工具。

---

**💡 Tip**: 本项目专注于内网环境的 Vibe Coding 体验优化，如果你的环境可以访问外网，建议直接使用[原版 Aider](https://github.com/Aider-AI/aider)获得最佳体验。