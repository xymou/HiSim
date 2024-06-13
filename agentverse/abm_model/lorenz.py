"""
Lorenz(2021) Model
"""
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


class LorenzAgent(mesa.Agent):
    """
    Lorenz's Model
    includes assimilation force, reinforcement force, similarity bias, polarization factor, source credibility
    """

    def __init__(self, model, unique_id, name, init_att, alpha=0.5, 
                 lamda=1, k=1, M=1, tho=0.5):
        """
        """
        
        super().__init__(unique_id, model)
        
        self.name = name
        # initial attitude
        self.att =  [init_att]
        # strength of the social influence
        self.alpha = alpha
        # hyperparameters
        self.lamda = lamda
        self.k = k 
        self.M = M
        self.tho = tho
        

    def step(self):
        """
        Assimilation Force: asm(a_it, m_jt) = m_jt-a_it
        Reinforcement Force: ref(m_jt) = m_jt
        Similarity Bias: sim(a_it, m_jt) = lambda^k/(lambda^k+abs(m_jt-a_it)^k) in [0,1], lambda and k specify the shape of sim bias function
        Other Assumptions: 
            alpha_i = alpha for all i;
            source credibility s(i,j) in [0,1]
            polarization factor pol(a_it) = (M^2-a_it^2)/M^2 in [0,1]
        Selection Fuction: one random agent in system
        Message Function: m_jt = a_jt
        Attitude update function: delta_a_it = alpha_i *s(i,j)*pol(a_it)*sim(a_it, m_jt)*[tho*asm(a_it, m_jt)+(1-tho)*ref(m_jt)]
            tho: the degree of assimilation(versus the reinforcemnet force), in [0,1]
            M: the theoretical boundary for the attitude space, a in [-M, M]
        """
        # attitude update
        att = self.att[-1]
        att_update = 0
        candidate_agents = []
        for agent in self.model.schedule.agents:
            if agent !=self:candidate_agents.append(agent)
        target_agent = random.choice(candidate_agents)
        m_jt = target_agent.att[-1]
        sim = pow(self.lamda, self.k)/(pow(self.lamda, self.k)+pow(abs(m_jt-att), self.k))
        pol = (pow(self.M, 2)-pow(att, 2))/pow(self.M, 2)
        asm = m_jt-att
        ref = m_jt
        att_update = self.alpha*pol*sim*(self.tho*asm+(1-self.tho)*ref)
        
        att = att + att_update
        self.att.append(att)
#         print(self.name, att)


@abm_registry.register("lorenz")
class LorenzModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, agent_config_lst, order, alpha, lamda, k, M, tho, llm_agents_atts):
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
            a = LorenzAgent(self, agent_config_lst[i]['id'], agent_config_lst[i]['name'], agent_config_lst[i]['init_att'],
                            alpha=alpha, lamda=lamda, k=k, M=M, tho=tho)
            # Add the agent to the scheduler
            self.schedule.add(a)
            self.name2idx[agent_config_lst[i]['name']] = i
        assert list(self.name2idx.keys()) == [a.name for a in self.agents]
        assert self.num_agents == len(self.agents)
            
    def get_attitudes(self):
        atts = [a.att[-1] for a in self.agents]
        return atts
        
    def step(self):
        """Advance the model by one step."""

        # The model's step will go here for now this will call the step method of each agent and print the agent's unique_id
        self.schedule.step()
#         print(self._steps)
        for agent in self.llm_agents_atts:
            self.update_mirror(agent, self.llm_agents_atts[agent][self._steps-1])
        
        
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