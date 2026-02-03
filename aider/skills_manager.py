from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

"""
Skills integration for Aider4IntraLLM.

This implements a lightweight subset of the "AgentSkills" directory convention used in
`software-agent-sdk-main` examples:
 - Each skill lives in a directory: <skills_root>/<skill-name>/
 - Each skill directory contains a `SKILL.md` file with YAML frontmatter:
     ---
     name: brainstorming
     description: Use when ...
     triggers: [optional]
     ---
 - Optional resource directories: scripts/, references/, assets/

We intentionally avoid importing `openhands.*` directly because the full SDK brings in extra
dependencies that are not part of this repo's runtime requirements. Instead, we reuse the same
file/folder conventions and frontmatter schema so skills from `superpowers-main/skills` and
AgentSkills-style repositories work out of the box.
"""


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    triggers: tuple[str, ...]
    skill_dir: Path
    skill_md: Path
    scripts: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()


def _find_skill_md(skill_dir: Path) -> Path | None:
    if not skill_dir.is_dir():
        return None
    for item in skill_dir.iterdir():
        if item.is_file() and item.name.lower() == "skill.md":
            return item
    return None


def _list_resource_files(resource_dir: Path) -> tuple[str, ...]:
    if not resource_dir.is_dir():
        return ()
    files: list[str] = []
    for item in resource_dir.rglob("*"):
        if item.is_file():
            files.append(str(item.relative_to(resource_dir)))
    return tuple(sorted(files))


def _parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown.

    Returns (metadata, body_without_frontmatter).
    If no frontmatter, metadata is {} and body is original text.
    """
    lines = md_text.splitlines(keepends=True)
    if not lines:
        return {}, md_text

    # Frontmatter must start at beginning.
    if lines[0].strip() != "---":
        return {}, md_text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, md_text

    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])
    try:
        meta = yaml.safe_load(fm_text) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}
    return meta, body


def _normalize_skill_name(name: str) -> str:
    # Support superpowers-style namespace prefix for convenience.
    name = (name or "").strip()
    if name.startswith("superpowers:"):
        name = name[len("superpowers:") :]
    if name.startswith("superpowers/"):
        name = name[len("superpowers/") :]
    return name.strip()


class SkillsManager:
    def __init__(self, roots: Iterable[str | Path], repo_root: str | Path | None = None):
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.roots = tuple(self._resolve_root(r) for r in roots if r)
        self._cache: dict[str, SkillEntry] | None = None

    def _resolve_root(self, root: str | Path) -> Path:
        p = Path(root)
        if not p.is_absolute() and self.repo_root:
            p = (self.repo_root / p).resolve()
        return p

    def discover(self, force: bool = False) -> dict[str, SkillEntry]:
        if self._cache is not None and not force:
            return self._cache

        skills: dict[str, SkillEntry] = {}
        for root in self.roots:
            if not root.exists() or not root.is_dir():
                continue
            for subdir in root.iterdir():
                if not subdir.is_dir():
                    continue
                skill_md = _find_skill_md(subdir)
                if not skill_md:
                    continue

                try:
                    text = skill_md.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                meta, _body = _parse_frontmatter(text)
                name = _normalize_skill_name(str(meta.get("name") or subdir.name))
                if not name or not SKILL_NAME_PATTERN.match(name):
                    # Keep non-conforming names accessible by directory name anyway.
                    name = _normalize_skill_name(subdir.name)
                description = str(meta.get("description") or "").strip()

                triggers_val = meta.get("triggers", None)
                trigger_val = meta.get("trigger", None)
                triggers: list[str] = []
                if isinstance(triggers_val, list):
                    triggers = [str(t).strip() for t in triggers_val if str(t).strip()]
                elif isinstance(triggers_val, str) and triggers_val.strip():
                    triggers = [triggers_val.strip()]
                elif isinstance(trigger_val, str) and trigger_val.strip():
                    triggers = [trigger_val.strip()]

                scripts = _list_resource_files(subdir / "scripts")
                references = _list_resource_files(subdir / "references")
                assets = _list_resource_files(subdir / "assets")

                skills[name] = SkillEntry(
                    name=name,
                    description=description,
                    triggers=tuple(triggers),
                    skill_dir=subdir,
                    skill_md=skill_md,
                    scripts=scripts,
                    references=references,
                    assets=assets,
                )

        self._cache = skills
        return skills

    def list(self) -> list[SkillEntry]:
        skills = self.discover()
        return sorted(skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> SkillEntry | None:
        name = _normalize_skill_name(name)
        if not name:
            return None
        skills = self.discover()
        if name in skills:
            return skills[name]
        # Fuzzy: allow "brainstorming/SKILL.md" or "brainstorming"
        name = name.split("/")[0]
        return skills.get(name)

