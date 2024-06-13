from agentverse.registry import Registry

llm_registry = Registry(name="LLMRegistry")
LOCAL_LLMS = [
    "llama-2-7b-chat-hf",
    "llama-2-13b-chat-hf",
    "vicuna-7b-v1.1",
    "vicuna-7b-v1.3",
    "vicuna-7b-v1.5",
    "vicuna-13b-v1.5",
    "mistral-7b-instruct-v0.1"
]
LOCAL_LLMS_MAPPING = {
    "llama-2-7b-chat-hf": "/remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/",
    "llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "vicuna-7b-v1.1": "/remote-home/share/LLM_CKPT/vicuna-7B-v1.1/",
    "vicuna-7b-v1.3": "/remote-home/share/LLM_CKPT/vicuna-7B-v1.3/",
    "vicuna-7b-v1.5": "/remote-home/share/LLM_CKPT/vicuna-7b-v1.5/",
    "vicuna-13b-v1.5": "/remote-home/share/LLM_CKPT/vicuna-13b-v1.5",
    "mistral-7b-instruct-v0.1":"/remote-home/share/LLM_CKPT/Mistral-7B-Instruct-v0.1/",
}

from .base import BaseLLM, BaseChatModel, BaseCompletionModel, LLMResult
from .openai import OpenAIChat
