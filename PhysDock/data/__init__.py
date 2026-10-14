"""PhysDock data pipeline.

* features.py   feature schema (FEATURES: shape / dtype / meaning of every model input)
* ccd.py        CCD library (reference conformers of chemical components)
* featurizer.py system pickle -> features (FeatureConfig, SystemEntry, SystemFeaturizer)
* meta.py       SystemMeta (atom names etc., PDB writing)
* dataloader.py PhysDockDataset, PhysDockCollator, PhysDockDataloader
"""
from PhysDock.data.features import FEATURES, OPTIONAL_FEATURES, FeatureSpec, validate_features
from PhysDock.data.ccd import CCDConformer, CCDLibrary, default_ccd_path
from PhysDock.data.meta import PDB_CHAIN_IDS, SystemMeta
from PhysDock.data.featurizer import (FeatureConfig, MissingMSAError, Sample, SystemEntry, SystemFeaturizer,
                                      msa_feat_for_round)
from PhysDock.data.dataloader import PhysDockCollator, PhysDockDataloader, PhysDockDataset, seed_worker
