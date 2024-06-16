from __future__ import annotations

"""
An agent based upon Observation-Reflection architecture.
"""

from logging import getLogger

from abc import abstractmethod
from typing import List, Set, Union, NamedTuple, TYPE_CHECKING

from pydantic import BaseModel, Field, validator

from agentverse.llms import BaseLLM
from agentverse.memory import BaseMemory, ChatHistoryMemory, PersonalMemory
from agentverse.twitter_page import TwitterPage
from agentverse.info_box import InfoBox
from agentverse.message import Message, TwitterMessage
from agentverse.output_parser import OutputParser

from agentverse.message import Message
from agentverse.agents.base import BaseAgent
from agentverse.utils import label_stance

from datetime import datetime as dt
import datetime
from datetime import timedelta
from textblob import TextBlob

#from . import agent_registry
from string import Template

from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent

logger = getLogger(__file__)

if TYPE_CHECKING:
    from agentverse.environments.base import BaseEnvironment


@agent_registry.register("twitter")
class TwitterAgent(BaseAgent):
    async_mode: bool = (True,)
    current_time: str = (None,)
    environment: BaseEnvironment = None
    step_cnt: int = 0
    page: TwitterPage = None
    info_box: InfoBox = None
    personal_history: BaseMemory = None # Field(default_factory=PersonalMemory)
    retrieved_memory:str = ""
    data_collector = {}
    atts: list = []
    context_prompt_template: str = Field(default="")

    manipulated_memory: str = Field(
        default="", description="one fragment used in prompt construction"
    )

    @validator("current_time")
    def convert_str_to_dt(cls, current_time):
        if not isinstance(current_time, str):
            raise ValueError("current_time should be str")
        return dt.strptime(current_time, "%Y-%m-%d %H:%M:%S")


    def att2score(self, content): # use Textblob
        # label stance
        stance = label_stance(content, self.environment.target)
        if stance in ['Oppose']:
            sign=-1
        else:
            sign=1
        blob = TextBlob(content)
        score = blob.sentiment.polarity
        return sign * abs(score), stance     


    async def get_personal_experience(self):
        # retrieve and summarize personal experience
        if not self.personal_history.has_summary:
            await self.personal_history.summarize()


    def step(self, current_time: dt, env_description: str = "") -> Message:
        """
        Call this method at each time frame
        """
        self.current_time = current_time

        # reflection
        self.manipulated_memory = self.memory_manipulator.manipulate_memory()

        # retrieve event memory
        if len(self.memory.messages):
            questions = self._fill_context_for_retrieval(env_description)
            questions = questions.split('\n')
            self.retrieved_memory = self.memory_manipulator.retrieve(questions, 10)
        else:
            self.retrieved_memory = ""

        prompt = self._fill_prompt_template(env_description)

        parsed_response, reaction, target, parent_id = None, None, None, None
        for i in range(self.max_retry):
            try:
                response = self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)

                if "post(" in parsed_response.return_values["output"]:
                    reaction = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'post'
                elif "retweet(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'retweet'
                elif "reply(" in parsed_response.return_values["output"]:
                    reaction, target, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )      
                    reaction_type = 'reply'
                elif "like(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )    
                    reaction_type = 'like'          
                elif "do_nothing(" in parsed_response.return_values["output"]:
                    reaction, target, reaction_type = None, None, None
                else:
                    raise Exception(
                        f"no valid parsed_response detected, "
                        f"cur response {parsed_response.return_values['output']}"
                    )
                break

            except Exception as e:
                logger.error(e)
                logger.warn("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        if reaction is None:
            reaction = "Silence"
            reaction_type = 'other'
            att = 0
            stance = 'Neutral'
        else:
            att, stance = self.att2score(reaction)

        message = TwitterMessage(
            content="" if reaction is None else reaction,
            post_time = current_time,
            msg_type = reaction_type,
            sender = self.name,
            receiver = self.get_receiver()
            if target is None
            else self.get_valid_receiver(target),
            parent_id = parent_id,
        )

        # add the news to the agent's memory
        if env_description!="":
            msg = Message(content=env_description, sender='News')
            self.add_message_to_memory([msg])
        self.atts.append(att)
        self.step_cnt += 1
        
        # save to output dict
        self.data_collector[self.environment.cnt_turn]={}
        self.data_collector[self.environment.cnt_turn]['prompt']=prompt
        self.data_collector[self.environment.cnt_turn]['response']=message
        self.data_collector[self.environment.cnt_turn]['parsed_response']=parsed_response
        self.data_collector[self.environment.cnt_turn]['att']=list(self.atts)
        self.data_collector[self.environment.cnt_turn]['stance'] = stance
        
        return message

    async def astep(self, current_time: dt, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        # use environment's time to update agent's time
        self.current_time = current_time

        # reflection      
        self.manipulated_memory = self.memory_manipulator.manipulate_memory()

        # retrieve event memory
        if len(self.memory.messages):
            questions = self._fill_context_for_retrieval(env_description)
            questions = questions.split('\n')
            self.retrieved_memory = self.memory_manipulator.retrieve(questions, 10)
        else:
            self.retrieved_memory = ""


        prompt = self._fill_prompt_template(env_description)

        parsed_response, reaction, target, parent_id = None, None, None, None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)

                if "post(" in parsed_response.return_values["output"]:
                    reaction = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'post'
                elif "retweet(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'retweet'
                elif "reply(" in parsed_response.return_values["output"]:
                    reaction, target, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )      
                    reaction_type = 'reply'
                elif "like(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )    
                    reaction_type = 'like'          
                elif "do_nothing(" in parsed_response.return_values["output"]:
                    reaction, target, reaction_type = None, None, None
                else:
                    raise Exception(
                        f"no valid parsed_response detected, "
                        f"cur response {parsed_response.return_values['output']}"
                    )
                break

            except Exception as e:
                logger.error(e)
                logger.warn("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        if reaction is None:
            reaction = "Silence"
            reaction_type = 'other'
            att = 0
            stance = 'Neutral'
        else:
            att, stance = self.att2score(reaction)

        message = TwitterMessage(
            content="" if reaction is None else reaction,
            post_time = current_time,
            msg_type = reaction_type,
            sender = self.name,
            receiver = self.get_receiver()
            if target is None
            else self.get_valid_receiver(target),
            parent_id = parent_id,
        )

        # add the news to the agent's memory
        if env_description!="":
            msg = Message(content=env_description, sender='News')
            self.add_message_to_memory([msg])
        self.atts.append(att)
        self.step_cnt += 1


        # save to output dict
        self.data_collector[self.environment.cnt_turn]={}
        self.data_collector[self.environment.cnt_turn]['prompt']=prompt
        self.data_collector[self.environment.cnt_turn]['response']=message
        self.data_collector[self.environment.cnt_turn]['parsed_response']=parsed_response
        self.data_collector[self.environment.cnt_turn]['att']=list(self.atts)
        self.data_collector[self.environment.cnt_turn]['stance'] = stance
        return message


    def _post(self, content=None):
        if content is None:
            return ""
        else:
            # reaction_content = (
            #     f"{self.name} posts a tweet: '{content}'."
            # )
            reaction_content = content
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content

    def _retweet(self, content=None, author=None, original_tweet_id=None, original_tweet=None):
        if author is None or original_tweet_id is None: return ""
        try:
            original_tweet = self.environment.tweet_db[original_tweet_id].content
        except:
            # raise Exception("Retweet. Not legal tweet id: {}".format(original_tweet_id))
            logger.warn("Retweet. Not legal tweet id: {}".format(original_tweet_id))
        # original_tweet = original_tweet_id
        if content is None:
            # reaction_content = (
            #         f"{self.name} retweets a tweet of [{author}]: '{original_tweet}'."
            #     )
            reaction_content = (
                    f"Retweets a tweet of [{author}]: '{original_tweet}'."
                )
        else:
            # reaction_content = (
            #     f"{self.name} retweets a tweet of [{author}]: '{original_tweet}' with additional statements: {content}."
            # )
            reaction_content = (
                f"Retweets a tweet of [{author}]: '{original_tweet}' with additional statements: {content}."
            )
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, original_tweet_id

    def _reply(self, content=None, author=None, original_tweet_id=None):
        if content is None or author is None or original_tweet_id is None: return ""
        try:
            original_tweet = self.environment.tweet_db[original_tweet_id].content
        except:
            # raise Exception("Comment. Not legal tweet id: {}".format(original_tweet_id))
            logger.warn("Reply. Not legal tweet id: {}".format(original_tweet_id))
        reaction_content = (
            f"{self.name} replies to [{author}]: {content}."
        )
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, author, original_tweet_id

    def _like(self, author=None, original_tweet_id=None):
        if author is None or original_tweet_id is None: return ""
        try:
            original_tweet = self.environment.tweet_db[original_tweet_id].content
        except:
            # raise Exception("Like. Not legal tweet id: {}".format(original_tweet_id))
            logger.warn("Like. Not legal tweet id: {}".format(original_tweet_id))
        reaction_content = f"{self.name} likes a tweet of [{author}]: '{original_tweet}'."

        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, original_tweet_id

    def get_valid_receiver(self, target: str) -> set():
        all_agents_name = []
        for agent in self.environment.agents:
            all_agents_name.append(agent.name)

        if not (target in all_agents_name):
            return {"all"}
        else:
            return {target}

    def _fill_prompt_template(self, env_description: str = "") -> str:
        """Fill the placeholders in the prompt template

        In the twitter agent, these placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${personal_history}: the personal experience (tweets) of the user
        - ${chat_history}: the chat history (about this movement) of the agent
        - ${tweet_page}: the tweet page the agent can see
        - ${trigger_news}: desc of the trigger event
        - ${info_box}: replies in notifications
        """
        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "personal_history": self.personal_history.summary,
            # "chat_history": self.memory.to_string(add_sender_prefix=True),
            "chat_history":self.retrieved_memory,
            "current_time": self.current_time,
            "trigger_news": env_description,
            "tweet_page":self.page.to_string() if self.page else "",
            "info_box": self.info_box.to_string()
        }
        return Template(self.prompt_template).safe_substitute(input_arguments)
    
    def _fill_context_for_retrieval(self, env_description: str = "") -> str:
        """Fill the placeholders in the prompt template

        In the twitter agent, these placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${personal_history}: the personal experience (tweets) of the user
        - ${chat_history}: the chat history (about this movement) of the agent
        - ${tweet_page}: the tweet page the agent can see
        - ${trigger_news}: desc of the trigger event
        - ${info_box}: replies in notifications
        """
        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "personal_history": self.personal_history.summary,
            "current_time": self.current_time,
            "trigger_news": env_description,
            "tweet_page":self.page.to_string() if self.page else "",
            "info_box": self.info_box.to_string()
        }
        return Template(self.context_prompt_template).safe_substitute(input_arguments)        

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def add_message_to_tweet_page(self, messages: List[Message]) -> None:
        self.page.add_message(messages)

    def add_message_to_info_box(self, messages: List[Message]) -> None:
        self.info_box.add_message(messages)

    def reset(self, environment: BaseEnvironment) -> None:
        """Reset the agent"""
        self.environment = environment
        self.memory.reset()
        self.memory_manipulator.agent = self
        self.memory_manipulator.memory = self.memory


    async def acontext_test(self, context_config) -> Message:
        """Test the agent given a specific context"""

        self.manipulated_memory = self.memory_manipulator.manipulate_memory()
        if not self.personal_history.has_summary:
            await self.personal_history.summarize()

        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "personal_history": self.personal_history.to_string(add_sender_prefix=True),
            "chat_history": self.memory.to_string(add_sender_prefix=True),
            "current_time": context_config.pop("current_time", ""),
            "trigger_news": context_config.pop("trigger_news", ""),
            "tweet_page":context_config.pop("tweet_page", ""),
            "info_box": context_config.pop("info_box", ""),
        }
        prompt = Template(self.prompt_template).safe_substitute(input_arguments)

        parsed_response, reaction, target, parent_id = None, None, None, None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)

                if "post(" in parsed_response.return_values["output"]:
                    reaction = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'post'
                elif "retweet(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'retweet'
                elif "reply(" in parsed_response.return_values["output"]:
                    reaction, target, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )      
                    reaction_type = 'reply'
                elif "like(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )    
                    reaction_type = 'like'          
                elif "do_nothing(" in parsed_response.return_values["output"]:
                    reaction, target, reaction_type = None, None, None
                else:
                    raise Exception(
                        f"no valid parsed_response detected, "
                        f"cur response {parsed_response.return_values['output']}"
                    )
                break

            except Exception as e:
                logger.error(e)
                logger.warn("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        if reaction is None:
            reaction = "Silence"
            reaction_type = 'other'
            att = 0
            stance = 'Neutral'
        else:
            att, stance = self.att2score(reaction)

        message = TwitterMessage(
            content="" if reaction is None else reaction,
            post_time = self.current_time,
            msg_type = reaction_type,
            sender = self.name,
            receiver = self.get_receiver()
            if target is None
            else self.get_valid_receiver(target),
            parent_id = parent_id,
        )
        parsed_response = {'response':parsed_response, "attitude":att, "stance":stance}
        return prompt, message, parsed_response