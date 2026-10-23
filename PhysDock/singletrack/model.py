"""The single-track model with PhysDock's interface: Pomelo's condition embedder and training / sampling (PhysDock.pomelo.model.Pomelo), the DiT replaced by SingleTrackDiT."""
import torch.nn as nn

from PhysDock.pomelo.embedders import ConditionEmbedder
from PhysDock.pomelo.model import Pomelo
from PhysDock.singletrack.configs import preset
from PhysDock.singletrack.dit import SingleTrackDiT


class SingleTrack(Pomelo):
    """SingleTrack(**model_config): PhysDock's model config (configs/model/singletrack_mini.yaml) plus `singletrack`: a preset of PhysDock.singletrack.configs or a dict"""

    def __init__(self, num_augmentation_sample: int, diffusion: dict, sigma_data: float, singletrack="mini", **physdock_only) -> None:
        nn.Module.__init__(self)
        from PhysDock.models.diffusion import build_diffuser
        self.cfg = preset(singletrack)
        self.num_augmentation_sample, self.sigma_data = num_augmentation_sample, sigma_data
        self.diffuser = build_diffuser(diffusion)
        self.diffusion_conditioning = ConditionEmbedder(self.cfg)
        self.dit = SingleTrackDiT(self.cfg)
