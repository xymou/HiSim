from pydantic import BaseModel, Field
from typing import List, Tuple, Set, Union, Any

from agentverse.utils import AgentAction
from datetime import datetime as dt


class Message(BaseModel):
    content: Any = Field(default="")
    sender: str = Field(default="")
    receiver: Set[str] = Field(default=set())
    sender_agent: object = Field(default=None)
    tool_response: List[Tuple[AgentAction, str]] = Field(default=[])


class SolverMessage(Message):
    pass


class CriticMessage(Message):
    is_agree: bool
    criticism: str = ""


class ExecutorMessage(Message):
    tool_name: str = Field(default="")
    tool_input: Any = None


class EvaluatorMessage(Message):
    score: Union[bool, List[bool], int, List[int]]
    advice: str = Field(default="")


class RoleAssignerMessage(Message):
    pass

class TwitterMessage(Message):
    post_time: dt
    msg_type: str='other'
    tweet_id: str=None
    parent_id: str=None
    num_rt: int=0
    num_cmt: int=0
    num_like: int=0
    receiver: Set[str] = Field(default=set())
    embedding:list=[]
    need_embedding: bool=True