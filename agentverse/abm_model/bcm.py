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


class BCAgent(mesa.Agent):
    """
    Deffuant's model is a BC model where N_J=1
    It assumes that if the message 𝑚𝑗,𝑡 is close enough to the agent 𝑖’s attitude 𝑎𝑖,𝑡 , 
    the message has an assimilation force on the agent’s attitude. 
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, bc_bound = 0.2):
        """
        bc_bound: the confidence bound
        """
        
        super().__init__(unique_id, model)
        
        self.name = name
        # initial attitude
        self.att =  [init_att]
        # strength of the social influence
        self.alpha = alpha
        self.bc_bound = bc_bound
        

    def step(self):
        """
        Selection Function: one random agent j in the system within the confidence bound
        Message Function: m_jt = a_jt
        Assimilation Force: asm(a_it, m_jt) = (m_jt-a_it)
        Similarity Bias: sim(a_it, m_jt) = 1 if diff< bc_bound, else 0
        """
        # attitude update
        att = self.att[-1]
        att_update = 0
        candidate_agents = []
        for agent in self.model.schedule.agents:
            # exclude the agent itself
            if agent == self:continue
            if abs(att-agent.att[-1])<self.bc_bound:
                candidate_agents.append(agent)
        # randomly sample
        if len(candidate_agents):
            target_agent = random.choice(candidate_agents)
            sim = 1
            att_update = target_agent.att[-1]-att
        else:
            sim = 0 
            att_update = 0
        att = att + self.alpha * att_update
        self.att.append(att)
#         print(self.name, att)



@abm_registry.register("bcm")
class BCModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order = 'concurrent', alpha=0.1, bc_bound=0.1,llm_agents_atts=[]):
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
        self.name2idx = {}
        for i in range(self.num_agents):
            a = BCAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'],
                         alpha=alpha, bc_bound = bc_bound)
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)

    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
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