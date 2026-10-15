from PhysDock.models.model import PhysDock
from PhysDock.models.loss import PhysDockLoss
from PhysDock.configs import compose, model_config
from PhysDock.utils.precision import set_matmul_precision

# TF32 by default: "high" is a no-op on GPUs without TF32, so this does not need to initialise CUDA at import time
# (scripts call set_matmul_precision(cfg.matmul_precision) again, "auto" checks the device)
set_matmul_precision("high")
