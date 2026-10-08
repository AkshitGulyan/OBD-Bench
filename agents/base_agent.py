from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    def __init__(self, agent_id: str, model: str, llm_client: Any | None = None, system_prompt: str | None = None):
        self.agent_id = agent_id
        self.model = model
        self.llm_client = llm_client
        self.system_prompt = system_prompt or "You are a helpful agent."

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any]):
        raise NotImplementedError

    async def call_llm(self, task: str, context: dict[str, Any], user_message: str | None = None):
        if self.llm_client is None:
            return {
                "agent_id": self.agent_id,
                "task": task,
                "response": user_message or task,
            }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message or str({"task": task, "context": context})},
        ]
        return await self.llm_client.generate(messages=messages, model=self.model)
