"""
Personal Experience of Twitter Users
- constructed from the user's historical tweets
"""
from typing import List, Union

from pydantic import Field

from agentverse.message import Message, TwitterMessage
from agentverse.llms import BaseLLM
from agentverse.llms.openai import get_embedding, OpenAIChat


from . import memory_registry
from .base import BaseMemory
from tqdm import tqdm
import json
import os
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import numpy as np
import openai

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def build_bm25_retriever(corpus):
    tokenized_corpus = []
    for passage in corpus:
        tokenized_corpus.append(bm25_tokenizer(passage))

    bm25 = BM25Okapi(tokenized_corpus)   
    return bm25      

def bm25_retrieve_facts(bm25, passages, ques, top_k=10, thred=0.8):
    bm25_scores = bm25.get_scores(bm25_tokenizer(ques))
    top_k = min(top_k, len(passages))
    top_n = np.argpartition(bm25_scores, -top_k)[-top_k:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    res = []
    idx = []
    for hit in bm25_hits:
        if hit['score']>=thred:
            res.append(passages[hit['corpus_id']].replace("\n", " "))
            idx.append(hit['corpus_id'])
    return res, idx 


@memory_registry.register("personal_history")
class PersonalMemory(BaseMemory):
    messages: List[Message] = Field(default=[])
    memory_path: str = None
    target: str = "the Metoo Movement"
    top_k: str = 5
    deadline: str = None
    model: str = "gpt-3.5-turbo"
    has_summary: bool = False
    max_summary_length: int = 200
    summary: str = ""
    SUMMARIZATION_PROMPT = '''Your task is to create a concise running summary of observations in the provided text, focusing on key and potentially important information to remember.

Please avoid repeating the observations and pay attention to the person's overall leanings. Keep the summary concise in one sentence.

Observations:
"""
{new_events}
"""
'''
    RETRIVEAL_QUERY='''What is your opinion on {target} or other political and social issues?'''

    def __init__(self, memory_path, target, top_k, deadline, llm):
        super().__init__()
        self.memory_path = memory_path
        self.target = target
        self.top_k = top_k
        self.deadline = deadline
        self.model = llm
        # load the historical tweets of the user
        if self.memory_path is not None and os.path.exists(self.memory_path):
            print('load ',self.memory_path)
            df = open(self.memory_path,'r',errors='ignore').readlines()
            content_set = set()
            for d in df:
                d = json.loads(d)
                content = d["rawContent"]
                content = re.sub(r"\n+", "\n", content)
                content.replace('\n',' ')
                if content in content_set or len(content.split())<10:continue
                content_set.add(content)
                post_time = d["date"][:19]
                if post_time>self.deadline:continue
                sender = d["user"]["username"]
                if sender !=self.memory_path.split('/')[-1][:-4]:continue
                message = TwitterMessage(content=content, post_time=post_time, sender=sender)
                self.messages.append(message)
            self.messages = self.bm25_retrieve(self.messages)
        else:
            print(self.memory_path,' does not exist!')

    def add_message(self, messages: List[Message]) -> None:
        for message in messages:
            self.messages.append(message)

    def reset(self) -> None:
        self.messages = []

    def bm25_retrieve(self, messages):
        if len(messages)==0:return []
        texts = [message.content for message in self.messages]
        query = self.RETRIVEAL_QUERY.format(target=self.target)
        bm25_retriever = build_bm25_retriever(texts)
        _, idx = bm25_retrieve_facts(bm25_retriever, texts, query, self.top_k)
        messages = [messages[i] for i in idx]
        return messages
 

    async def summarize(self):
        self.has_summary = True
        messages = self.messages
        print('summarize personal experience:', len(messages))
        if len(messages)==0:
            self.summary=''
            return
        texts = self.to_string(add_sender_prefix=True)
        prompt = self.SUMMARIZATION_PROMPT.format(
            new_events=texts
        )
        response = await openai.ChatCompletion.acreate(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_summary_length,
            temperature=0.5,
        )
        self.summary =  response["choices"][0]["message"]["content"]   
        message = Message(content=self.summary)
        self.add_message([message])
        
    def to_string(self, add_sender_prefix: bool = False) -> str:   
        if add_sender_prefix:
            return "\n".join(
                [
                    f"[{message.sender}] posted a tweet: {message.content}"
                    if message.sender != ""
                    else message.content
                    for message in self.messages
                ]
            )
        else:
            return "\n".join([message.content for message in self.messages])