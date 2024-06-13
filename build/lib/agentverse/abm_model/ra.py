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


class RAAgent(mesa.Agent):
    """
    Deffuant's 2002 Model
    Relative Agreement Model
    
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, init_uct=0.4):
        """
        
        """
        
        super().__init__(unique_id, model)
        
        self.name = name
        # initial attitude
        self.att =  [init_att]
        # strength of the social influence
        self.alpha = alpha
        # uncertainty item
        self.uct = [init_uct]
        

    def step(self):
        """
        Attitude Update Function: \delta = alpha * sim(a_itm u_itm a_jt, u_jt)*(a_it-a_jt)
        sim(a_it, u_it, a_jt, u_jt) = (h_ij/u_j)-1, if (h_ij/u_j)>1 else 0 
        h_ij = min(a_it,+u_it, a_jt+u_jt)-max(a_it-u_it, a_jt-u_jt),    (h_ij/u_j)-1 is referred to as the relative agreement
        the uncertainty term u_it is assumed to update in a similar way
        - Adssimilation Force: asm(a_it, m_jt) = (m_jt-a_it)
        - Similarity Bias: sim(a_it, u_it, a_jt, u_jt) = (h_ij/u_j)-1, if (h_ij/u_j)>1 else 0 
        - other assumptions: agent i's attitude is modeled as a segment [a_it-u_it, a_it+u_it], msg also as a segment [a_jt-u_jt, a_jt+u_jt]
        - Selection Function: one random agent in system except the agent i itself
        - Message Function: m_jt = [a_jt-u_jt, a_jt+u_jt]
        """
        # attitude update
        att = self.att[-1]
        uct = self.uct[-1]
        att_update = 0
        
        candidate_agents = []
        for agent in self.model.schedule.agents:
            # exclude the agent itself
            if agent == self:continue
            candidate_agents.append(agent)
        
        target_agent = random.choice(candidate_agents)
        h_ij = min(att+uct, target_agent.att[-1]+target_agent.uct[-1])-max(att-uct, target_agent.att[-1]-target_agent.uct[-1])
        if h_ij/target_agent.uct[-1]>1:
            sim = h_ij/target_agent.uct[-1]-1
        else:
            sim = 0
        
        # how to update uct, the same with a_it (according to the paper)
        att_update = sim * (target_agent.att[-1]-att)
        att = att + self.alpha * att_update
        self.att.append(att)
        uct_update = sim * (target_agent.uct[-1]-uct)
        uct = uct + self.alpha * uct_update
        self.uct.append(uct)
#         print(self.name, att, uct)

@abm_registry.register("ra")
class RAModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order = 'concurrent',alpha=0.5, init_uct=0.4,llm_agents_atts=[]):
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
        name2id = {}
        for i in range(self.num_agents):
            a = RAAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'],
                        alpha=alpha, init_uct=init_uct)
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)
        # Selection Function: a random agent in the system

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