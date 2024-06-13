from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

from agentverse.environments.simulation_env.rules.describer import (
    BaseDescriber,
    describer_registry,
)
from agentverse.environments.simulation_env.rules.order import BaseOrder, order_registry
from agentverse.environments.simulation_env.rules.selector import (
    BaseSelector,
    selector_registry,
)
from agentverse.environments.simulation_env.rules.updater import (
    BaseUpdater,
    updater_registry,
)
from agentverse.environments.simulation_env.rules.visibility import (
    BaseVisibility,
    visibility_registry,
)
from agentverse.environments import BaseRule

if TYPE_CHECKING:
    from agentverse.environments.base import BaseEnvironment

from agentverse.message import Message, TwitterMessage
from agentverse.environments.simulation_env.rules import SimulationRule

class TwitterRule(SimulationRule):
    """
    Rule for the environment. It controls the speaking order of the agents
    and maintain the set of visible agents for each agent.
    """

    def update_tweet_page(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the tweet page of the agent who is able to see that message"""
        self.updater.update_tweet_page(environment, *args, **kwargs)

    def update_info_box(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the tweet page of the agent who is able to see that message"""
        self.updater.update_info_box(environment, *args, **kwargs)

    def update_memory(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the memory of the agent who is able to see that message"""
        self.updater.update_memory(environment, *args, **kwargs)

    def update_tweet_db(self, environment: BaseEnvironment, *args, **kwargs):
        messages = environment.last_messages
        for m in messages:
            if isinstance(m, TwitterMessage) and m.msg_type == 'post':
                idx = str(len(environment.tweet_db))
                environment.tweet_db[idx] = m
                m.tweet_id = str(idx)
            # update num_rt of original tweet
            elif isinstance(m, TwitterMessage) and m.msg_type == 'retweet':
                idx = str(len(environment.tweet_db))
                environment.tweet_db[idx] = m
                m.tweet_id = str(idx)                
                idx = m.parent_id
                if idx in environment.tweet_db:
                    environment.tweet_db[idx].num_rt+=1
            # update num_cmt of original tweet
            elif isinstance(m, TwitterMessage) and m.msg_type == 'comment':
                idx = m.parent_id
                environment.tweet_db[idx].num_cmt+=1
            # update num_like of original tweet
            elif isinstance(m, TwitterMessage) and m.msg_type == 'like':
                idx = m.parent_id
                environment.tweet_db[idx].num_like+=1

    def update_tweet_db_for_news(self, environment, author, content):
        idx = str(len(environment.tweet_db))
        m = TwitterMessage(
            content=content,
            sender=author, 
            receiver=set({"all"}),
            post_time=environment.current_time,
            msg_type='post',
            tweet_id=idx,
            parent_id=None,
            num_rt=0,
            num_cmt=0,
            num_like=0,         
        )
        environment.tweet_db[idx] = m    
        return [m]   

    def update_tweet_page_for_news(self, environment: BaseEnvironment,msg_lst) -> None:
        """For each message, add it to the tweet page of the agent who is able to see that message"""
        self.updater.update_tweet_page_for_news(environment, msg_lst)