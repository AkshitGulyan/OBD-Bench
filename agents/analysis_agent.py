from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    def __init__(self, agent_id: str = "analysis_agent", model: str = "gpt-4o-mini", llm_client: Any | None = None):
        super().__init__(
            agent_id=agent_id,
            model=model,
            llm_client=llm_client,
            system_prompt=(
                "You are the analysis agent. Convert the task data into a clear logical interpretation, "
                "extract the core constraints, and identify the required intermediate outputs."
            ),
        )

    async def execute(self, task: str, context: dict[str, Any]):
        return await self.call_llm(task, context, user_message=f"Analyze the findings and produce a concise reasoning summary: {task}")
