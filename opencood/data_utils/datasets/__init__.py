from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
try:
    from opencood.data_utils.datasets.rawdataset import \
        RawEarlyFusionDataset as RawEarlyFusionDataset
except ImportError:
    RawEarlyFusionDataset = None

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset
}
if RawEarlyFusionDataset is not None:
    __all__['RawEarlyFusionDataset'] = RawEarlyFusionDataset

# the final range for evaluation
GT_RANGE = [-100, -40, -5, 100, 40, 3]
# The communication range for cavs
COM_RANGE = 70

def build_dataset(dataset_cfg, visualize=False, train=True, isSim=None,
                  ego_only=None, nofusion=None, single_pc=None,
                  cav_label_only=None):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in __all__, error_message

    isSim = dataset_cfg.get('isSim', False) if isSim is None else isSim
    ego_only = dataset_cfg.get('ego_only', False) if ego_only is None else ego_only
    nofusion = dataset_cfg.get('nofusion', False) if nofusion is None else nofusion
    single_pc = dataset_cfg.get('single_pc', False) \
        if single_pc is None else single_pc
    cav_label_only = dataset_cfg.get('cav_label_only', False) \
        if cav_label_only is None else cav_label_only

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        isSim=isSim,
        ego_only=ego_only,
        nofusion=nofusion,
        single_pc=single_pc,
        cav_label_only=cav_label_only
    )

    return dataset
