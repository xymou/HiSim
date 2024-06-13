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

page_registry = Registry(name="PageRegistry")

@page_registry.register("timeline")
class TwitterPage(BaseMemory):
    """
    messages: list of TwitterMessage
    tweet_num: illustrate most recent tweet_num tweets
    """
    messages: List[Message] = Field(default=[])
    tweet_num: int=5
    
    def add_message(self, messages: List[Message]) -> None:
        # only reserve the post/retweet action
        # pay attention to the storage size
        for message in messages:
            self.messages.insert(0, message)
        self.messages = self.messages[:self.tweet_num]

    def to_string(self) -> str:
        return "\n".join(
            [
                f'tweet id: {message.tweet_id} [{message.sender}]: {message.content} --Post Time: {message.post_time}'
                for message in self.messages
            ]
        )

    def reset(self) -> None:
        self.messages = []
