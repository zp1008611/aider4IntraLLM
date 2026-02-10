# 使用技巧


## agent

## architect


## code

## ask



## skills 

### skills 是什么

本项目支持 **AgentSkills 风格**的 skills：每个 skill 是一个文件夹，包含 `SKILL.md`（可选再带脚本/参考资料等）。你可以把常用工作流（调试、写计划、做 Code Review、写文档等）沉淀成 skill，在需要时一键加载到对话只读上下文中。

skills 默认从仓库内 `skills/` 扫描。也可以通过启动参数追加更多 skills 根目录。

### 常用命令（在 Aider 交互界面里输入，以 `/` 开头）

- **列出 skills**
  - `/skill list`
  - `/skill list <关键词>`（按名称/描述过滤）
- **搜索 skills**
  - `/skill search <关键词>`
- **查看 skill 信息**
  - `/skill show <skill-name>`（查看描述、路径、资源文件列表等）
- **加载 skill 到上下文**
  - `/skill load <skill-name>`（把该 skill 的 `SKILL.md` 加入只读上下文）
- **查看扫描根目录**
  - `/skill roots`

> 注意：`/skill ...` 只能在 Aider 的交互提示符里使用（例如出现 `>` 后）。不要在 PowerShell 直接输入 `/skill`。

### 实战技巧（推荐用法）

- **先 list / search，再 load**
  - 不确定有没有现成流程时先 `/skill search debug`、`/skill list plan`，再按需 `/skill load ...`。
- **遇到复杂任务先加载“流程型” skill**
  - 例如要写实施计划：先 `/skill load writing-plans`
  - 遇到 bug/测试失败：先 `/skill load systematic-debugging`
  - 要做 TDD：先 `/skill load test-driven-development`
- **把“可复用模板/规范”放进 skill**
  - 比如周报/月报模板、PR 模板、提交信息规范、代码审查清单等，写成一个 skill，之后复用成本最低。

### 如何新增自己的 skill（项目内维护）

1. 在仓库根目录创建：`skills/<your-skill-name>/SKILL.md`
2. `SKILL.md` 顶部写 YAML frontmatter（至少包含 `name`、`description`）：

```markdown
---
name: my-skill
description: Use when 需要生成月报/周报，或需要固定的结构化输出模板
---

# My Skill

这里写你的流程、模板、注意事项……
```

3. 进入 Aider 后执行：`/skill load my-skill`

### 指定额外 skills 目录（可选）

如果你想把 skills 放在仓库外（例如公司共享目录），启动时追加：

```powershell
python -m aider --skills-dir D:\path\to\skills --skills-dir D:\more\skills
```
