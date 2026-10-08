from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    def __init__(self, agent_id: str = "critic_agent", model: str = "gpt-4o-mini", llm_client: Any | None = None):
        super().__init__(
            agent_id=agent_id,
            model=model,
            llm_client=llm_client,
            system_prompt=(
                "You are the critic agent. Validate the reasoning, surface inconsistencies, and verify that the output "
                "meets the required constraints before finalization."
            ),
        )

    async def execute(self, task: str, context: dict[str, Any]):
        return await self.call_llm(task, context, user_message=f"Critique the draft answer and identify any gaps or errors: {task}")
