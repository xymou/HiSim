from __future__ import annotations

import random
import json
from typing import TYPE_CHECKING, Any, List, Union, Dict

from . import visibility_registry as VisibilityRegistry
from .base import BaseVisibility

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@VisibilityRegistry.register("twitter")
class TwitterVisibility(BaseVisibility):
    """
    Visibility function for twitter: each agent can only see his or her following list

    Args:

        following_info:
            The follower list information. If it is a string, then it should be a
            path of json file storing the following info. If it is a
            dict of list of str, then it should be the following information of each agent.
    """

    follower_info: Union[str, Dict[str, List[str]]]
    current_turn: int = 0

    def update_visible_agents(self, environment: BaseEnvironment):
        self.update_receiver(environment)


    def update_receiver(self, environment: BaseEnvironment, reset=False):
        if self.follower_info is None:
            for agent in environment.agents:
                agent.set_receiver(set({agent.name})) # can only see itself
        else:
            if isinstance(self.follower_info, str):
                groups = json.load(open(self.follower_info, 'r'))
            else:
                groups = self.follower_info
            for agent in environment.agents:
                if agent.name in groups:
                    # add the agent itself
                    fl_list = groups[agent.name]+[agent.name]
                    agent.set_receiver(set(fl_list))
                else:
                    # can only see itself
                    agent.set_receiver(set({agent.name}))

    def reset(self):
        self.current_turn = 0
