import json
import logging
import os
import openai
import copy
from typing import List, Optional, Tuple, Dict

from pydantic import Field

from agentverse.message import Message, TwitterMessage
from agentverse.memory import BaseMemory
from agentverse.llms.utils import count_message_tokens, count_string_tokens
from agentverse.llms import OpenAIChat

from agentverse.registry import Registry

info_registry = Registry(name="InfoRegistry")

@info_registry.register("basic")
class InfoBox(BaseMemory):
    """
    messages: list of TwitterMessage
    cmt_num: illustrate most recent tweet_num comments
    tweet_num: illustrate most recent tweet_num tweets
    """
    messages: Dict[str, List[Message]] = Field(default={})
    cmt_num: int=10
    tweet_num: int=3
    
    def add_message(self, messages: List[Message]) -> None:
        # store comments into different groups (group by parent id)
        for message in messages:
            parent_id = message.parent_id
            if parent_id not in self.messages:
                self.messages[parent_id] = []
            self.messages[parent_id].insert(0, message)
            self.messages[parent_id] = self.messages[parent_id][:self.tweet_num]

    def to_string(self) -> str:
        return "\n".join(["original tweet id:"+parent_id+":\n"+"\n".join(
            [
                f'[{message.sender}]: {message.content}'
                for message in self.messages[parent_id]
            ])
            for parent_id in self.messages
            ]
        )

    def reset(self) -> None:
        self.messages = {}
