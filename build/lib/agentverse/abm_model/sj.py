import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt
from agentverse.abm_model import abm_registry
from agentverse.logging import get_logger

logger = get_logger()


class SJAgent(mesa.Agent):
    """
    Jager & Amblard (2005)’s Model (Social Judgement Model)
    includes a repulsion force based on the BC model (not only acceptance but also rejection)
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, acc_thred=0.5, rej_thred=0.5):
        """
        
        """
        
        super().__init__(unique_id, model)
        
        self.name = name
        # initial attitude
        self.att =  [init_att]
        # strength of the social influence
        self.alpha = alpha
        # threds of acceptance and rejection
        self.acc_thred = acc_thred
        self.rej_thred = rej_thred
        

    def step(self):
        """
        Selection Function: one random agent in the system
        Message Fuction: m_jt = a_jt
        Assimilation Force: asm(a_it, m_jt) = (m_jt-a_it)
        Similarity Bias: sim(a_it, a_jt) = 1 if abs(a_jt-a_it)<u_i, u_i is the threshold of acceptance of agent i
        Repulsion Force: rep(a_it, a_jt) = -(a_jt-a_it) if abs(a_jt-a_it)>t_i, t_i is the threshold for the rejection of the agent i
        Other Assumptions: alpha in [0,1]; each agent has two thresholds
        
        """
        # attitude update
        att = self.att[-1]
        att_update = 0 
        candidate_agents = []
        for agent in self.model.schedule.agents:
            if agent !=self:candidate_agents.append(agent)
        target_agent = random.choice(candidate_agents)
        sim = (target_agent.att[-1]-att) if abs(target_agent.att[-1]-att)<self.acc_thred else 0
        rep = -(target_agent.att[-1]-att) if abs(target_agent.att[-1]-att)>self.rej_thred else 0
        att_update = sim*(target_agent.att[-1]-att)+rep
        
        att = att + self.alpha * att_update
        self.att.append(att)
#         print(self.name, att)


@abm_registry.register("sj")
class SJModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order = 'concurrent', alpha=0.5, acc_thred=0.5, rej_thred=0.8, llm_agents_atts=[]):
        super().__init__()
        self.num_agents = len(agent_config_lst)
        self.llm_agents_atts = llm_agents_atts
        self.name2idx = {}
        # Create scheduler and assign it to the model
        if order =='concurrent':
            self.schedule = mesa.time.BaseScheduler(self)
        elif order =='simultaneous':
            self.schedule = mesa.time.SimultaneousActivation(self)
        elif order =='random':
            self.schedule = mesa.time.RandomActivation(self)
        else:
            raise NotImplementedError

        # Create agents
        for i in range(self.num_agents):
            a = SJAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'],
                             alpha=alpha, acc_thred = acc_thred, rej_thred = rej_thred)
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)      

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
#         print(self._steps)
        for agent in self.llm_agents_atts:
            self.update_mirror(agent, self.llm_agents_atts[agent][self._steps-1])

    def get_attitudes(self):
        atts = [a.att[-1] for a in self.agents]
        return atts
    
    def get_measures(self, target_attitudes,ne_att=0):
        """
        target_attitudes: empirical data
        output measures: bias, diversity
        - bias: the deviation of the mean attitude from the neutral attitude
        - diversity: the standard deviation of attitudes
        """
        simu_atts = self.get_attitudes()
        
        # empirical
        bias = np.mean(target_attitudes)-ne_att
        diversity = np.var(target_attitudes)
        
        # simu
        simu_bias = np.mean(simu_atts)-ne_att
        simu_diversity = np.var(simu_atts)
        
        delta_bias = abs(simu_bias-bias)
        delta_diversity = abs(simu_diversity-diversity)

        return {'bias':bias,
               'diversity':diversity,
               'simu_bias':simu_bias,
               'simu_diversity':simu_diversity,
               'delta_bias':delta_bias,
               'delta_diversity':delta_diversity}
               
    def update_mirror(self, name, att):
        idx = self.name2idx[name]
        self.agents[idx].att[-1] = att