from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from . import describer_registry as DescriberRegistry
from .base import BaseDescriber

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@DescriberRegistry.register("twitter")
class TwitterDescriber(BaseDescriber):
    
    def get_env_description(self, environment: BaseEnvironment) -> List[str]:
        """Return the environment description for each agent"""
        cnt_turn = environment.cnt_turn
        trigger_news = environment.trigger_news
        if cnt_turn in trigger_news:
            # broadcast the event news
            return [trigger_news[cnt_turn] for _ in range(len(environment.agents))]
        else:
            return ["" for _ in range(len(environment.agents))]

    def reset(self) -> None:
        pass