## Skills 目录（Aider4IntraLLM）

这里存放本项目使用的 **AgentSkills 风格** skills（目录结构为 `<skill-name>/SKILL.md`）。

### 目录约定

- 每个 skill 一个文件夹：`skills/<skill-name>/`
- 每个 skill 必须有：`skills/<skill-name>/SKILL.md`
- 可选资源目录：
  - `scripts/`：可执行脚本
  - `references/`：参考资料
  - `assets/`：静态资源

### 如何新增自己的 skill

1. 新建目录：`skills/my-skill/`
2. 新建 `skills/my-skill/SKILL.md`，包含 YAML frontmatter：

```markdown
---
name: my-skill
description: Use when [触发条件/使用场景]
---

# My Skill

[你的 skill 内容...]
```

3. 进入 Aider 会话后使用：
   - `/skill list`
   - `/skill load my-skill`

### 来源说明

本目录初始内容由 [superpowers-main/skills/](https://github.com/obra/superpowers) 迁移而来，便于后续在项目内持续新增和维护 skills。

