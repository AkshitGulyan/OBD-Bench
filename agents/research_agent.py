from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    def __init__(self, agent_id: str = "research_agent", model: str = "gpt-4o-mini", llm_client: Any | None = None):
        super().__init__(
            agent_id=agent_id,
            model=model,
            llm_client=llm_client,
            system_prompt=(
                "You are the research agent. Gather the relevant facts, identify the key inputs, "
                "and return a structured summary of the task context."
            ),
        )

    async def execute(self, task: str, context: dict[str, Any]):
        return await self.call_llm(task, context, user_message=f"Research the task and summarize the key facts: {task}")
