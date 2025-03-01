import os
import torch
import logging
import os.path as osp
import laspy
from pathlib import Path
from typing import List, Tuple
from itertools import product
import re

from src.datasets import BaseDataset
from src.data import Data
from src.datasets.fractal_config import *


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with fractal on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['FRACTAL', 'MiniFRACTAL']


########################################################################
#                                 Utils                                #
########################################################################

def read_fractal_tile(
        filepath: str,
        xyz: bool = True,
        intensity: bool = True,
        semantic: bool = True,
        rgb: bool = True,
        remap: bool = False
) -> Data:
    """Read a fractal tile saved as LAZ.

    :param filepath: str
        Absolute path to the LAZ file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param intensity: bool
        Whether intensity should be saved in the output Data.intensity
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their fractal ID
        to their train ID.
    """
    data = Data()
    
    # Read LAZ file using laspy
    las = laspy.read(filepath)
    if xyz:
        pos = torch.tensor(las.xyz, dtype=torch.float)
        pos_offset = pos[0]
        data.pos = pos - pos_offset
        data.pos_offset = pos_offset

    if intensity:
        # Normalize intensity to [0, 1] range
        data.intensity = torch.tensor(las.intensity.astype(np.int16), dtype=torch.float).clip(min=0, max=60000) / 60000

    if semantic:
        # Assuming semantic labels are stored in classification field
        y = torch.tensor(las.classification, dtype=torch.long)
        if remap:
            # Map unknown classes to 0 (Unclassified)
            remapped_y = torch.zeros_like(y)
            for orig_id, train_id in MAP_ID_TO_TRAINID.items():
                remapped_y[y == orig_id] = train_id
            data.y = remapped_y
        else:
            data.y = y

    if rgb:
        data.rgb = torch.tensor(las.color, dtype=torch.float)

    return data


########################################################################
#                                FRACTAL                                #
########################################################################

class FRACTAL(BaseDataset):
    """fractal dataset.

    Dataset website: https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}
    transform : `callable`
        transform function operating on data.
    pre_transform : `callable`
        pre_transform function operating on data.
    pre_filter : `callable`
        pre_filter function operating on data.
    on_device_transform: `callable`
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    def __init__(self, *args, **kwargs):
        # Store the stage before super().__init__
        self.all_base_cloud_ids_dict = None
        self.all_cloud_ids_dict = None
        self._stage = kwargs.get('stage', 'train')
        
        # Temporarily store a flag to skip check_cloud_ids during parent initialization
        self._skip_cloud_check = True
        super().__init__(*args, **kwargs)
        self._skip_cloud_check = False
        
        # Now that initialization is complete, we can check cloud IDs
        self.check_cloud_ids()

    def check_cloud_ids(self) -> None:
        """Override parent's check_cloud_ids to skip during initialization"""
        if getattr(self, '_skip_cloud_check', False):
            return
        super().check_cloud_ids()

    @property
    def class_names(self) -> List[str]:
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return FRACTAL_NUM_CLASSES


    @property
    def class_colors(self) -> List[List[int]]:
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS


    @property
    def all_cloud_ids(self) -> dict:
        """Dictionary holding lists of clouds ids, for each stage."""
        # If clouds are tiled, expand and append all cloud names with a
        # suffix indicating which tile it corresponds to
        if self.xy_tiling is not None:
            tx, ty = self.xy_tiling if isinstance(self.xy_tiling, tuple) else (self.xy_tiling, self.xy_tiling)
            if self.all_cloud_ids_dict is None:
                self.all_cloud_ids_dict = {
                    stage: [
                        f'{ci}__TILE_{x + 1}-{y + 1}_OF_{tx}-{ty}'
                    for ci in ids
                    for x, y in product(range(tx), range(ty))]
                for stage, ids in self.all_base_cloud_ids.items()}
        return self.all_cloud_ids_dict

    @property
    def all_base_cloud_ids(self) -> dict:
        """Dictionary holding lists of base clouds ids, for each stage."""
        if self.all_base_cloud_ids_dict is None:
            train = [Path(f).stem for f in Path(self.raw_dir, 'train').glob('*.laz')][:500]
            val = [Path(f).stem for f in Path(self.raw_dir, 'val').glob('*.laz')][:50]
            test = [Path(f).stem for f in Path(self.raw_dir, 'test').glob('*.laz')][:50]
            self.all_base_cloud_ids_dict = {'train': train, 'val': val, 'test': test}
        return self.all_base_cloud_ids_dict


    def read_single_raw_cloud(self, raw_cloud_path: str) -> 'Data':
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        return read_fractal_tile(
            raw_cloud_path, intensity=True, semantic=True, rgb=True, remap=True)

    def download_dataset(self) -> None:
        return None
    

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            └── {{train, test, val}}/
                └── {{tile_name}}.laz
            """

    def id_to_base_id(self, id: str) -> str:
        """Given an ID, remove the tiling indications, if any."""
        # If no tiling pattern is found, return the ID as is
        tile_info = self.get_tile_from_path(id)
        if tile_info is None:
            return id
        return tile_info[1]

    def id_to_relative_raw_path(self, id: str) -> str:
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        if id in self.all_cloud_ids['train']:
            stage = 'train'
        elif id in self.all_cloud_ids['val']:
            stage = 'val'  # Changed: val files are in their own directory
        elif id in self.all_cloud_ids['test']:
            stage = 'test'
        else:
            raise ValueError(f"Unknown tile id '{id}'")
        
        # Get base ID without tiling information
        base_id = self.id_to_base_id(id)
        return osp.join(stage, base_id + '.laz')

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]

        # Keep the original stage directory structure
        # Removed: stage = 'train' if stage in ['trainval', 'val'] else stage

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = osp.join(self.raw_dir, stage, base_cloud_id + '.laz')

        return raw_path

    @staticmethod
    def get_tile_from_path(path: str) -> Tuple[Tuple, str, str]:
        """Override the parent's get_tile_from_path to handle our specific tiling format."""
        # Search the XY tiling suffix pattern
        out_reg = re.search('__TILE_(\d+)-(\d+)_OF_(\d+)-(\d+)', path)
        if out_reg is not None:
            x, y, x_tiling, y_tiling = [int(g) for g in out_reg.groups()]
            suffix = f'__TILE_{x}-{y}_OF_{x_tiling}-{y_tiling}'
            prefix = path.replace(suffix, '')
            return (x - 1, y - 1, (x_tiling, y_tiling)), prefix, suffix
        return None

    @property
    def raw_file_names(self) -> List[str]:
        """The file paths to find in order to skip the download."""
        if self.raw_file_names_3d_dict is None:
            self.raw_file_names_3d_dict = {
                stage: [self.id_to_relative_raw_path(x) for x in self.all_cloud_ids[stage]]
                for stage in self.all_cloud_ids.keys()
            }
        # Flatten the dictionary values into a single list
        return [path for paths in self.raw_file_names_3d_dict.values() for path in paths]

    @property
    def stuff_classes(self) -> List[int]:
        """List of 'stuff' labels for instance and panoptic segmentation.
        In this dataset, all classes are "stuff" since we don't have instance
        segmentation.
        """
        return list(range(self.num_classes))


########################################################################
#                              MiniFRACTAL                              #
########################################################################

class MiniFRACTAL(FRACTAL):
    """A mini version of fractal with only a few windows for
    experimentation.
    """
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self) -> dict:
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self) -> None:
        super().process()

