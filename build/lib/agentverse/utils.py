from typing import NamedTuple, Union
from enum import Enum

import abc
import re
import openai
from tenacity import retry, stop_after_attempt, wait_exponential


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class AgentCriticism(NamedTuple):
    """Agent's criticism."""

    is_agree: bool
    criticism: str
    sender_agent: object = None


class AGENT_TYPES(Enum):
    ROLE_ASSIGNMENT = 0
    SOLVER = 1
    CRITIC = 2
    EXECUTION = 3
    EVALUATION = 4
    MANAGER = 5

    @staticmethod
    def from_string(agent_type: str):
        str_to_enum_dict = {
            "role_assigner": AGENT_TYPES.ROLE_ASSIGNMENT,
            "solver": AGENT_TYPES.SOLVER,
            "critic": AGENT_TYPES.CRITIC,
            "executor": AGENT_TYPES.EXECUTION,
            "evaluator": AGENT_TYPES.EVALUATION,
            "manager": AGENT_TYPES.MANAGER,
        }
        assert (
            agent_type in str_to_enum_dict
        ), f"Unknown agent type: {agent_type}. Check your config file."
        return str_to_enum_dict.get(agent_type.lower())


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def judge_response(response, labels):
    if response.strip() in labels:
        return response.strip()
    for label in labels:
        if label in response:
            return label
    return ''

def label_stance(text, target):
    pro_patterns={
        'Metoo Movement':['#Withyou',],
        'Black Lives Matter Movement':['#BlackLivesMatter', '#GeorgeFloyd', '#PoliceBrutality', '#BLM',],
        'the protection of Abortion Rights':['#roevwadeprotest', 'roe v wade protest', 'pro choice', 'pro-choice', 
                 '#prochoice', '#forcedbirth', 'forced birth', '#AbortionRightsAreHumanRights', 
                 'abortion rights Are Human Rights', '#MyBodyMyChoice', 'My Body My Choice', 
                 '#AbortionisHealthcare', 'abortion is healthcare', 'AbortionIsAHumanRight', 
                 'abortion is a human right', 'ReproductiveHealth', 'Reproductive Health', 
                 'AbortionRights', 'abortion rights' ]}
    for w in pro_patterns[target]:
        if w.lower() in text.lower():
            return 'Support'
    con_patterns = {
        'Metoo Movement':[],
        'Black Lives Matter Movement':[],        
        'the protection of Abortion Rights': ['#prolife', '#EndAbortion',
                    '#AbortionIsMurder', '#LifeIsAHumanRight', '#ChooseLife',
                    '#SaveTheBabyHumans', '#ValueLife', '#RescueThePreborn', '#EndRoeVWade',
                    '#MakeAbortionUnthinkable','#LiveActionAmbassador','#AbortionIsNotARight', '#AbortionIsRacist']}
    for w in con_patterns[target]:
        if w.lower() in text.lower():
            return 'Oppose'
    prompt = "What's the author's stance on {}? Please choose from Support, Neutral, and Oppose. Only output your choice.\n\n".format(target)
    text_sample = "Text: "+text+'\n'+'Stance: '
    prompt = prompt+text_sample
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "user", "content": prompt}
        ],
        temperature = 0,
        max_tokens = 16,
        )
    response = completion.choices[0].message.content
    ans = judge_response(response, ['Support', 'Neutral','Oppose'])
    return ans   



