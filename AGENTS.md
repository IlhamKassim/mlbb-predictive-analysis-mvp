# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Gemini CLI, Cursor, etc.) when working with this repository.

## Repository Overview
MLBB Predictive Analysis MVP - A Python project for predictive modeling and hero recommendations in Mobile Legends: Bang Bang.

## Agent Skills Integration
This repository has integrated [agent-skills](https://github.com/addyosmani/agent-skills).

### Core Rules
- If a task matches a skill, you **MUST** invoke it.
- Skills are installed in `.gemini/skills/` (for Gemini CLI) or can be referenced from the `agent-skills` repository.
- Never implement directly if a skill applies.
- Always follow the skill instructions exactly.

### Intent → Skill Mapping
- **Feature / new functionality** → `spec-driven-development` -> `planning-and-task-breakdown` -> `incremental-implementation` -> `test-driven-development`
- **Bug / failure** → `debugging-and-error-recovery` -> `test-driven-development`
- **Refactoring** → `code-simplification`
- **Code Review** → `code-review-and-quality`

### Anti-Rationalization
- "This is too small for a skill" - **Incorrect**.
- "I can just quickly implement this" - **Incorrect**.
- "I'll add tests later" - **Incorrect**.

**Always use skills to ensure production-grade quality.**
