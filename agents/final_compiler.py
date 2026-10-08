from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class FinalCompiler(BaseAgent):
    def __init__(self, agent_id: str = "final_compiler", model: str = "gpt-4o-mini", llm_client: Any | None = None):
        super().__init__(
            agent_id=agent_id,
            model=model,
            llm_client=llm_client,
            system_prompt=(
                "You are the final compiler. Combine the intermediate results, ensure all required fields are present, "
                "and produce the final answer in a structured format."
            ),
        )

    async def execute(self, task: str, context: dict[str, Any]):
        return await self.call_llm(task, context, user_message=f"Compile the final answer from the provided results: {task}")
