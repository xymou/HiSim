from __future__ import annotations

from typing import TYPE_CHECKING, List
import random

from . import order_registry as OrderRegistry
from .base import BaseOrder

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@OrderRegistry.register("twitter")
class TwitterOrder(BaseOrder):
    """
    The agents speak concurrently in a random order
    """

    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        res = list(range(len(environment.agents)))
        random.shuffle(res)
        return res
