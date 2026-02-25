"""Agent factory for prompt template loading flows."""

from pathlib import Path

from afk.agents import Agent

from .fake_llm import PromptAwareLLM


def build_agent(project_root: Path) -> Agent:
    return Agent(
        model=PromptAwareLLM(),
        name="OperationsBrief",
        instruction_file="operations_prompt.md",
        prompts_dir=project_root / "prompts",
    )
