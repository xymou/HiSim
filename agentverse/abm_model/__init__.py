from agentverse.registry import Registry

abm_registry = Registry(name="AbmRegistry")

from agentverse.abm_model.lorenz import LorenzModel
from agentverse.abm_model.bcm import BCModel
from agentverse.abm_model.bcmm import BCMultiModel
from agentverse.abm_model.sj import SJModel
from agentverse.abm_model.ra import RAModel
