# 🙌HiSim: A Hybrid Social Media Simulation Framework

This repository is the implementation of our ACL 2024-Findings paper [Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation](https://arxiv.org/abs/2402.16333).   
The framework simulates social media users in different ways:   
- For LLM-empowered core users, the implementation is build on [AgentVerse](https://github.com/OpenBMB/AgentVerse), many thanks to THUNLP for the open-source resource.
- For ordinary users supported by conventional ABMs, we use the [mesa](https://mesa.readthedocs.io/en/stable/) library to implement the agent-based models such as the Bounded Confidence Model.


## Contents
- [Abstract](#Abstract)
- [Dataset](#Dataset)
- [Getting Started](#Getting-Started)
- [Citation](#Citation)

## Abstract 
Social media has emerged as a cornerstone of social movements, wielding significant influence in driving societal change. Simulating the response of the public and forecasting the potential impact has become increasingly important. However, existing methods for simulating such phenomena encounter challenges concerning their efficacy and efficiency in capturing the behaviors of social movement participants. In this paper, we introduce a hybrid framework **HiSim** for social media user simulation, wherein users are categorized into two types. Core users are driven by Large Language Models, while numerous ordinary users are modeled by deductive agent-based models. We further construct a Twitter-like environment to replicate their response dynamics following trigger events. Subsequently, we develop a multi-faceted benchmark SoMoSiMu-Bench for evaluation and conduct comprehensive experiments across real-world datasets. Experimental results demonstrate the effectiveness and flexibility of our method.


## Dataset
To be **in compliance with Twitter’s terms of service**, we can not publish the raw data. Instead, we only disclose the original tweet ids, from which you can filter out the users you want to study, to minimize the privacy risk.  
- <i>Metoo</i>: from [#metoo Digital Media Collection](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2SRSKJ), we further keep the tweets during the events, where the ids can be downloaded at [metoo_link](https://drive.google.com/file/d/1qQzQAvDH-eLtg1jPTKe6NkToF7Aq1EAA/view?usp=sharing).
- <i>Roe</i>: from [#RoeOverturned](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/STU0J5&version=1.2), we further keep the tweets during the events, where the ids can be downloaded at [roe_link](https://drive.google.com/file/d/13dkJ_P2JzbrDdJkYdwred260Ps-ym-64/view?usp=sharing).
- <i>BLM</i>: from [blm_twitter_corpus](https://github.com/sjgiorgi/blm_twitter_corpus), we further keep the tweets during the events, where the ids can be downloaded at [blm_link](https://drive.google.com/file/d/1HymVETg5SgLJqL1O3bPiT-RcBVSMGEhT/view?usp=sharing).

For user list we used in our paper, we can only provide the ids [id_link](https://drive.google.com/drive/folders/1AaOFZQ0NSwzoeLFdDubE95Kdv3vBHoji?usp=sharing).

## Getting Started
### Installation
```bash
conda create -n HiSim python=3.9
conda activate HiSim
git clone https://github.com/xymou/HiSim.git
cd HiSim
pip install -e .
```

### Environment Variables
You need to export your OpenAI API key as follows：
```bash
# Export your OpenAI API key
export OPENAI_API_BASE="your_api_base_here"
export OPENAI_API_KEY="your_api_key_here"
```

### Simulation
#### Framework Required Modules
```
- agentverse 
  - agents
    - simulation_agent
      - twitter
  - environments
    - simulation_env
      - twitter
  - abm_model
  - twitter_page
  - info_box
  - message
```

#### Micro (individual) User Behavior Replication
The micro-level simulation aims to simulate the behaviors of users at the individual level given a certain context in the pattern of single-round simulation. In this scenarios, we do not include multi-agent interaction but only observe the replication of individual behaviors.
Here is an example:
```shell
agentverse-microtest --task simulation/roe_micro --ckpt /remote-home/xymou/xymou_page/HiSim/ckpt/roe_micro/
```
For micro-level simulation, you mainly need to prepare the agent list and the corresponding context list in the [config.yaml](https://github.com/xymou/HiSim/blob/main/agentverse/tasks/simulation/roe_micro/config.yaml). Since a user can be involved in different (user, context) tuples, so there may be repeated agent in the list. Data fields in the "context" includes:
- tweet_page: the real tweet that user can see
- trigger_news: the offline event news at the corresponding time
- text: the ground truth reponse of the user
- msg_type: the message type of the ground truth reponse
Note that text and msg_type will not be used in simulation. They are provided for subsequent evaluation.

#### Macro (system) Opinion Dynamics
The macro-level simulation runs for consecutive rounds, to help observe how collective opinions shift over time resulting from agent interactions.
Here is an example:
```shell
agentverse-simulation --task simulation/roe_macro_hybrid --ckpt /remote-home/xymou/xymou_page/HiSim/ckpt/roe_macro/
```

You can create your own social media simulation by defining new scenarios in agentverse/tasks/simulation. There are some key points to define a macro-level simulation:
- Social Network Cnonstruction: you can assign the social networks of agents by uploading their follower list, either give a dict like {"userA":["followerA", "followerB"]}, or give a file path to "follower_info" of visibility, as the example in * example_data/follower_dict.json * .
- Offline News Feed: pre-defined offline news can be fed by "trigger_news" in environment. It is a dict whose key is the turn of news feeding and the value is the content of the news.
- Conventional ABM-driven Users: you need to provide the type and parameters of the abm models and initial attitudes of all the agents (including both the core users and oridinary users) in the "abm_model" field
- Target: the target/topic of the opinion modeling, such as "the Protection of Abortion", "Metoo Movement"
- Personal Experience: you can provide the real historical tweets of the users to model the personal memory, where the txt file can be specified in memory_path of personal_history of agent. The format of the authentic user tweets can be found in * example_data/sample_user_tweets *. If the historical tweets are not available, you can set the path to None.  
A full example can be found in the [config.yaml](https://github.com/xymou/HiSim/blob/main/agentverse/tasks/simulation/roe_macro_hybrid/config.yaml)

Note:  
If you want to run the simulation with LLM-based agents only (instead of the hybrid pattern), just set the abm config to None. An exmaple can be found in the [config.yaml](https://github.com/xymou/HiSim/blob/main/agentverse/tasks/simulation/roe_macro_llm/config.yaml)
.


## Citation
Please consider citing this paper if you find this repository useful:
```bash
@article{mou2024unveiling,
      title={Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation}, 
      author={Xinyi Mou and Zhongyu Wei and Xuanjing Huang},
      year={2024},
      journal = {arXiv preprint arXiv: 2402.16333},
}
```