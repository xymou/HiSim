import asyncio
import logging
from typing import List
import pickle

# from agentverse.agents import Agent
from agentverse.agents.simulation_agent.conversation import BaseAgent
from agentverse.environments import BaseEnvironment
from agentverse.initialization import load_agent, load_environment, prepare_task_config, load_abm_model

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


class Simulation_Align:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment, context:List, ckpt_dir:str):
        self.agents = agents
        self.environment = environment
        self.context = context
        self.ckpt_dir = ckpt_dir

    @classmethod
    def from_task(cls, task: str, tasks_dir: str, ckpt_dir: str):
        """Build an AgentVerse from a task name.
        The task name should correspond to a directory in `tasks` directory.
        Then this method will load the configuration from the yaml file in that directory.
        """
        # Prepare the config of the task
        task_config = prepare_task_config(task, tasks_dir)

        # Build the agents
        agents = []
        for agent_configs in task_config["agents"]:
            agent = load_agent(agent_configs)
            agents.append(agent)
       
        # Build the environment
        env_config = task_config["environment"]
        env_config["agents"] = agents
        environment = load_environment(env_config)

        context = task_config["context"]

        return cls(agents, environment, context, ckpt_dir)

    @classmethod
    def from_ckpt(cls, ckpt_dir: str):
        agents = pickle.load(open(ckpt_dir+'agents.pkl', 'rb'))
        environment = pickle.load(open(ckpt_dir+'environment.pkl', 'rb'))
        context = pickle.load(open(ckpt_dir+'context.pkl', 'rb'))
        return cls(agents, environment, context, ckpt_dir)

    def run(self):
        """Run the environment from scratch until it is done."""
        """Initialize the environment for each sample"""
        data = []
        for i in range(len(self.agents)):
            agent= self.agents[i]
            self.environment.reset()
            prompt, message, parsed_response = asyncio.run(self.environment.test(agent, self.context[i]))
            data.append({'prompt':prompt, 'response':message, 'context':self.context[i], 'parsed_response':parsed_response})
            self.environment.report_metrics()
            self.save_ckpt()
            print('Output to {}'.format(self.environment.output_path))
            with open(self.environment.output_path,'wb') as f:
                pickle.dump(data, f)

    def save_ckpt(self):
        with open(self.ckpt_dir+'agents.pkl','wb') as f:
            pickle.dump(self.agents, f)
        with open(self.ckpt_dir+'environment.pkl','wb') as f:
            pickle.dump(self.environment, f)      
        with open(self.ckpt_dir+'context.pkl','wb') as f:
            pickle.dump(self.context, f)                    
        
    def run_from_ckpt(self):
        """Run the environment from ckpt."""
        data = pickle.load(open(self.environment.output_path,'rb'))
        for i in range(len(data), len(self.agents)):
            agent= self.agents[i]
            self.environment.reset()
            prompt, message, parsed_response = asyncio.run(self.environment.test(agent, self.context[i]))
            data.append({'prompt':prompt, 'response':message, 'context':self.context[i], 'parsed_response':parsed_response})
            self.environment.report_metrics()
            self.save_ckpt()
            print('Output to {}'.format(self.environment.output_path))
            with open(self.environment.output_path,'wb') as f:
                pickle.dump(data, f)        


    def reset(self):
        self.environment.reset()
        for agent in self.agents:
            agent.reset()

    def next(self, *args, **kwargs):
        """Run the environment for one step and return the return message."""
        return_message = asyncio.run(self.environment.step(*args, **kwargs))
        return return_message

    def update_state(self, *args, **kwargs):
        """Run the environment for one step and return the return message."""
        self.environment.update_state(*args, **kwargs)

