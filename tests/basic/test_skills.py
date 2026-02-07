import argparse
import os
from pathlib import Path
from unittest import TestCase, mock

from aider.coders import Coder
from aider.commands import Commands
from aider.io import InputOutput
from aider.models import Model
from aider.skills_manager import SkillsManager
from aider.utils import GitTemporaryDirectory


class TestSkillsManager(TestCase):
    def test_discover_parses_frontmatter_and_resources(self):
        with GitTemporaryDirectory() as repo_dir:
            repo_dir = Path(repo_dir)
            skills_root = repo_dir / "my-skills"
            skill_dir = skills_root / "rot13-encryption"
            (skill_dir / "scripts").mkdir(parents=True)
            (skill_dir / "references").mkdir(parents=True)

            (skill_dir / "scripts" / "encrypt.sh").write_text("#!/bin/sh\necho ok\n")
            (skill_dir / "references" / "examples.md").write_text("# Examples\n")

            (skill_dir / "SKILL.md").write_text(
                """---
name: rot13-encryption
description: Use when you need rot13
triggers:
  - encrypt
  - decrypt
---

# ROT13

Some content here.
""",
                encoding="utf-8",
            )

            mgr = SkillsManager(roots=[skills_root], repo_root=repo_dir)
            skills = mgr.discover()

            self.assertIn("rot13-encryption", skills)
            s = skills["rot13-encryption"]
            self.assertEqual(s.name, "rot13-encryption")
            self.assertEqual(s.description, "Use when you need rot13")
            self.assertEqual(set(s.triggers), {"encrypt", "decrypt"})
            self.assertIn("encrypt.sh", s.scripts)
            self.assertIn("examples.md", s.references)

    def test_discover_supports_trigger_singular_field(self):
        with GitTemporaryDirectory() as repo_dir:
            repo_dir = Path(repo_dir)
            skills_root = repo_dir / "my-skills"
            skill_dir = skills_root / "code-style-guide"
            skill_dir.mkdir(parents=True)

            (skill_dir / "SKILL.md").write_text(
                """---
name: code-style-guide
description: Use when writing code
trigger: style
---

# Style
""",
                encoding="utf-8",
            )

            mgr = SkillsManager(roots=[skills_root], repo_root=repo_dir)
            s = mgr.get("code-style-guide")
            self.assertIsNotNone(s)
            self.assertEqual(list(s.triggers), ["style"])


class TestSkillCommands(TestCase):
    def _make_coder_and_commands(self, repo_root: Path, skills_dir: Path | None = None):
        io = InputOutput(pretty=False, fancy_input=False, yes=True)
        model = Model("gpt-3.5-turbo")
        coder = Coder.create(model, None, io)

        # Ensure repo root is stable for skills path resolution.
        coder.root = str(repo_root.resolve())
        coder.abs_fnames = set()
        coder.abs_read_only_fnames = set()

        args = argparse.Namespace(skills_dir=[str(skills_dir)] if skills_dir else [])
        commands = Commands(io, coder, args=args)
        return io, coder, commands

    def test_cmd_skill_list_show_load(self):
        with GitTemporaryDirectory() as repo_dir:
            repo_root = Path(repo_dir)

            # Provide a default superpowers-style skills directory under repo root.
            skill_dir = repo_root / "superpowers-main" / "skills" / "brainstorming"
            skill_dir.mkdir(parents=True)
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: brainstorming
description: Use when you need design refinement
---

# Brainstorming
""",
                encoding="utf-8",
            )

            io, coder, commands = self._make_coder_and_commands(repo_root)

            out = []

            def capture(msg=""):
                out.append(str(msg))

            with mock.patch.object(io, "tool_output", side_effect=capture), mock.patch.object(
                io, "tool_warning", side_effect=capture
            ):
                commands.cmd_skill("list")
                self.assertTrue(any("brainstorming" in line for line in out))

                out.clear()
                commands.cmd_skill("show brainstorming")
                self.assertTrue(any("Skill: brainstorming" in line for line in out))
                self.assertTrue(any("SKILL.md" in line for line in out))

                out.clear()
                commands.cmd_skill("load brainstorming")
                abs_md = str(skill_md.resolve())
                self.assertIn(abs_md, coder.abs_read_only_fnames)
                self.assertTrue(any("已加载 skill" in line for line in out))

                # idempotent
                out.clear()
                commands.cmd_skill("load brainstorming")
                self.assertTrue(any("Skill 已在只读上下文中" in line for line in out))

    def test_cmd_skill_search_and_roots(self):
        with GitTemporaryDirectory() as repo_dir:
            repo_root = Path(repo_dir)

            extra_root = repo_root / "extra-skills"
            skill_dir = extra_root / "systematic-debugging"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: systematic-debugging
description: Use when debugging production issues
---
""",
                encoding="utf-8",
            )

            io, _coder, commands = self._make_coder_and_commands(repo_root, skills_dir=extra_root)

            out = []

            def capture(msg=""):
                out.append(str(msg))

            with mock.patch.object(io, "tool_output", side_effect=capture), mock.patch.object(
                io, "tool_warning", side_effect=capture
            ), mock.patch.object(io, "tool_error", side_effect=capture):
                commands.cmd_skill("roots")
                self.assertTrue(any("extra-skills" in line for line in out))

                out.clear()
                commands.cmd_skill("search debug")
                self.assertTrue(any("systematic-debugging" in line for line in out))

    def test_completions_skill_includes_subcommands_and_skill_names(self):
        with GitTemporaryDirectory() as repo_dir:
            repo_root = Path(repo_dir)
            extra_root = repo_root / "extra-skills"
            skill_dir = extra_root / "code-style-guide"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: code-style-guide
description: Use when writing code
---
""",
                encoding="utf-8",
            )

            _io, _coder, commands = self._make_coder_and_commands(repo_root, skills_dir=extra_root)
            cands = commands.completions_skill()

            self.assertIn("list", cands)
            self.assertIn("load", cands)
            self.assertIn("code-style-guide", cands)

