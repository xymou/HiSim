from __future__ import annotations

import os
from typing import Dict, List, TYPE_CHECKING

import yaml
from agentverse.logging import logger

try:
    from bmtools.agent.singletool import import_all_apis, load_single_tools
except:
    logger.warn(
        "BMTools is not installed, tools in the simulation environment cannot be used. To install BMTools, please follow the instruction in the README.md file. If you aren't running a *simulation* case with tool, you can ignore this warning."
    )

from agentverse.llms import llm_registry

from agentverse.agents import agent_registry
from agentverse.environments import BaseEnvironment, env_registry
from agentverse.memory import memory_registry
from agentverse.memory_manipulator import memory_manipulator_registry

from agentverse.output_parser import output_parser_registry
from agentverse.twitter_page import page_registry
from agentverse.info_box import info_registry
from agentverse.abm_model import abm_registry
import mesa

if TYPE_CHECKING:
    from agentverse.agents import BaseAgent


def load_llm(llm_config: Dict):
    llm_type = llm_config.pop("llm_type", "text-davinci-003")

    return llm_registry.build(llm_type, **llm_config)


def load_memory(memory_config: Dict, llm=None):
    memory_type = memory_config.pop("memory_type", "chat_history")
    if llm is not None:
        memory_config['llm'] = llm
    return memory_registry.build(memory_type, **memory_config)

def load_personal_history(memory_config: Dict):
    memory_type = memory_config.pop("memory_type", "personal_history")
    return memory_registry.build(memory_type, **memory_config)    

def load_memory_manipulator(memory_manipulator_config: Dict):
    memory_manipulator_type = memory_manipulator_config.pop(
        "memory_manipulator_type", "basic"
    )
    return memory_manipulator_registry.build(
        memory_manipulator_type, **memory_manipulator_config
    )

def load_page(page_config: Dict):
    if page_config is None: return None
    page_type = page_config.pop("page_type", "timeline")
    return page_registry.build(page_type, **page_config)

def load_info_box(info_box_config: Dict):
    if info_box_config is None: return None
    info_box_type = info_box_config.pop("info_box_type", "basic")
    return info_registry.build(info_box_type, **info_box_config)


def load_tools(tool_config: List[Dict]):
    if len(tool_config) == 0:
        return []
    all_tools_list = []
    for tool in tool_config:
        _, config = load_single_tools(tool["tool_name"], tool["tool_url"])
        all_tools_list += import_all_apis(config)
    return all_tools_list


def load_environment(env_config: Dict) -> BaseEnvironment:
    env_type = env_config.pop("env_type", "basic")
    return env_registry.build(env_type, **env_config)


def load_agent(agent_config: Dict) -> BaseAgent:
    agent_type = agent_config.pop("agent_type", "conversation")
    agent = agent_registry.build(agent_type, **agent_config)
    return agent

def load_abm_model(abm_config: Dict) -> mesa.Model:
    abm_model_type = abm_config.pop("model_type","lorenz")
    abm_model = abm_registry.build(abm_model_type, **abm_config)
    return abm_model


def prepare_task_config(task, tasks_dir):
    """Read the yaml config of the given task in `tasks` directory."""
    all_task_dir = tasks_dir
    task_path = os.path.join(all_task_dir, task)
    config_path = os.path.join(task_path, "config.yaml")
    if not os.path.exists(task_path):
        all_tasks = []
        for task in os.listdir(all_task_dir):
            if (
                os.path.isdir(os.path.join(all_task_dir, task))
                and task != "__pycache__"
            ):
                all_tasks.append(task)
                for subtask in os.listdir(os.path.join(all_task_dir, task)):
                    if (
                        os.path.isdir(os.path.join(all_task_dir, task, subtask))
                        and subtask != "__pycache__"
                    ):
                        all_tasks.append(f"{task}/{subtask}")
        raise ValueError(f"Task {task} not found. Available tasks: {all_tasks}")
    if not os.path.exists(config_path):
        raise ValueError(
            "You should include the config.yaml file in the task directory"
        )
    task_config = yaml.safe_load(open(config_path))

    for i, agent_configs in enumerate(task_config["agents"]):
        llm = load_llm(agent_configs.get("llm", "text-davinci-003"))
        agent_configs["llm"] = llm
        agent_configs["memory"] = load_memory(agent_configs.get("memory", {}), llm)
        if agent_configs.get("personal_history"):
            agent_configs['personal_history'] = load_personal_history(agent_configs.get("personal_history", {}))
        if agent_configs.get("tool_memory", None) is not None:
            agent_configs["tool_memory"] = load_memory(agent_configs["tool_memory"])

        page = load_page(agent_configs.get("page", None))
        agent_configs["page"] = page
        info_box = load_info_box(agent_configs.get("info_box", None))
        agent_configs['info_box'] = info_box

        memory_manipulator = load_memory_manipulator(
            agent_configs.get("memory_manipulator", {})
        )
        agent_configs["memory_manipulator"] = memory_manipulator

        agent_configs["tools"] = load_tools(agent_configs.get("tools", []))

        # Build the output parser
        output_parser_config = agent_configs.get("output_parser", {"type": "dummy"})
        if output_parser_config.get("type", None) == "role_assigner":
            output_parser_config["cnt_critic_agents"] = task_config.get(
                "cnt_critic_agents", 0
            )
        output_parser_name = output_parser_config.pop("type", task)
        agent_configs["output_parser"] = output_parser_registry.build(
            output_parser_name, **output_parser_config
        )

    return task_config
