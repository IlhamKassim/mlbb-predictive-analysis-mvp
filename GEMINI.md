# Project: MLBB Predictive Analysis MVP

This project focuses on predictive analysis for Mobile Legends: Bang Bang (MLBB), including match outcome prediction, hero recommendations, and player performance forecasting.

## Architecture
- `app.py`: Main entry point (likely a Flask or FastAPI app).
- `data_preprocessing.py`: Handles data cleaning and feature engineering.
- `model_training.py`: Training scripts for classifiers and regressors.
- `recommendation_system.py`: Logic for the hero recommendation engine.
- `data/`: CSV datasets.
- `models/`: Serialized model files (`.joblib`).

## Engineering Standards

### Agent Skills Mandate
This project uses the [Production-Grade Engineering Skills](https://github.com/addyosmani/agent-skills). These skills are installed in `.gemini/skills/` and are **mandatory** for all non-trivial tasks.

#### Core Workflows
- **New Features:** Use `spec-driven-development` -> `planning-and-task-breakdown` -> `incremental-implementation` + `test-driven-development`.
- **Bugs/Fixes:** Use `debugging-and-error-recovery` -> `test-driven-development`.
- **Refactoring:** Use `code-simplification` and `code-review-and-quality`.

#### Available Skills
You can discover and invoke skills using the `using-agent-skills` meta-skill.
Commonly used skills:
- `test-driven-development`: ALWAYS add tests for new logic.
- `spec-driven-development`: Document requirements before implementation.
- `planning-and-task-breakdown`: Break large tasks into manageable steps.
- `doubt-driven-development`: Use for high-stakes changes or complex logic.

### Local Conventions
- **Language:** Python 3.x
- **Testing:** (Verify existing test suite, e.g., pytest or unittest).
- **Documentation:** Use docstrings for all public functions and classes.
- **Model Management:** Use `joblib` for serialization.

## Task Lifecycle
1. **Research:** Map the codebase and validate assumptions.
2. **Strategy:** Define the approach using appropriate skills (Spec/Plan).
3. **Execution:** Implement incrementally with TDD.
4. **Validation:** Ensure all tests pass and linting is clean.
