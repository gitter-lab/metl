""" custom PyTorch Sampler for sampling batches grouped by PDB file """

from collections import defaultdict
from typing import List, Optional, Iterator
import logging

import numpy as np
from numpy.random import default_rng
from torch.utils.data import Sampler
import torch.distributed as dist

logger = logging.getLogger("METL." + __name__)
logger.setLevel(logging.DEBUG)


class PDBSampler(Sampler[List[int]]):
    """ Sample batches grouped by PDB file """
    def __init__(self, pdb_fns, batch_size, shuffle=False, drop_last=False):
        """ if drop_last is True, will drop the last non-full batch for each PDB file """
        super().__init__(data_source=None)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # build a mapping from pdb_fn to indices
        self.pdb_map = defaultdict(list)
        for i, pdb_fn in enumerate(pdb_fns):
            self.pdb_map[pdb_fn].append(i)

        # list of unique PDBs
        self.unique_pdb_fns = self.pdb_map.keys()

    def __iter__(self):
        # build up the batches using the sampling algorithm
        batches = []

        # shuffle the indices for each PDB in-place
        # ensures batches for each PDB are different every epoch
        if self.shuffle:
            for k, v in self.pdb_map.items():
                np.random.shuffle(v)

        # create batches for all the PDBs and append batches list
        for k, v in self.pdb_map.items():
            for batch_idx in range(0, len(v), self.batch_size):
                if self.drop_last and (batch_idx + self.batch_size > len(v)):
                    # drop the last batch if it is not full
                    continue
                batch = v[batch_idx:(batch_idx + self.batch_size)]
                batches.append(batch)

        # batches are currently in order of PDBs
        # shuffle to ensure random order of PDBs
        if self.shuffle:
            np.random.shuffle(batches)

        # finally return an iterator for this sampling of PDBs
        return iter(batches)

    def __len__(self):
        # calculate the number of batches depending on whether drop_last is true or false
        num_batches = 0
        for k, v in self.pdb_map.items():
            if self.drop_last:
                # compute the number of full batches
                num_batches += len(v) // self.batch_size
            else:
                # compute the total number of batches
                num_batches += (len(v) + self.batch_size - 1) // self.batch_size
        return num_batches


class PDBSamplerDistributed(Sampler[List[int]]):
    def __init__(self,
                 pdb_fns: List[int],
                 batch_size: int,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False):

        super().__init__(data_source=None)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.num_replicas = num_replicas
        self.rank = rank

        self.epoch = 0

        # build a mapping from pdb_fn to indices
        self.pdb_map = defaultdict(list)
        for i, pdb_fn in enumerate(pdb_fns):
            self.pdb_map[pdb_fn].append(i)

        # list of unique PDBs
        self.unique_pdb_fns = self.pdb_map.keys()
        # print("[RANK {}] There are {} unique pdb_fns".format(os.getenv("LOCAL_RANK", '0'),
        #                                                            len(self.unique_pdb_fns)))

    def __iter__(self) -> Iterator[List[int]]:
        # print("[RANK {}] PDBSamplerDistributed iterator (shuffle={}) is being initialized with epoch={}".format(
        #     os.getenv("LOCAL_RANK", '0'), self.shuffle, self.epoch))

        # build up the batches using the sampling algorithm
        batches = []

        # shuffle the indices for each PDB in-place
        # ensures batches for each PDB are different every epoch
        if self.shuffle:
            rng = default_rng(self.seed + self.epoch)
            for k, v in self.pdb_map.items():
                rng.shuffle(v)

        # create batches for all the PDBs and append batches list
        for k, v in self.pdb_map.items():
            for batch_idx in range(0, len(v), self.batch_size):
                if self.drop_last and (batch_idx + self.batch_size > len(v)):
                    # drop the last batch if it is not full
                    continue
                batch = v[batch_idx:(batch_idx + self.batch_size)]
                batches.append(batch)

        # batches are currently in order of PDBs
        # shuffle to ensure random order of PDBs
        if self.shuffle:
            rng.shuffle(batches)

        # ensure equal number of batches per GPU by dropping extra batches that aren't evenly divisible
        remainder = self._num_batches() % self.num_replicas
        if remainder != 0:
            # drop the remainder batches
            del batches[-remainder:]
        assert len(batches) % self.num_replicas == 0

        # grab just the batches for this rank
        # cleverly uses list slicing with starting index set to self.rank and stride set to self.num_replicas
        batches = batches[self.rank::self.num_replicas]
        assert len(batches) == self.__len__()
        # print("[RANK {}] PDBSamplerDistributed (shuffle={}) iterator contains {} batches".format(
        #     os.getenv("LOCAL_RANK", '0'), self.shuffle, len(batches)))

        # finally return an iterator for this sampling of PDBs
        return iter(batches)

    def _num_batches(self) -> int:
        # calculate the number of batches depending on whether drop_last is true or false
        # does not account for dropping batches to make batches evenly divisible by number of GPUs
        num_batches = 0
        for k, v in self.pdb_map.items():
            if self.drop_last:
                # compute the number of full batches
                num_batches += len(v) // self.batch_size
            else:
                # compute the total number of batches
                num_batches += (len(v) + self.batch_size - 1) // self.batch_size

        return num_batches

    def __len__(self) -> int:
        num_batches = self._num_batches()

        # account for strategy we use to make each GPU have the same number of batches
        # our strategy is to drop any extra batches that aren't evenly divisible
        # this differs from PyTorch's strategy of duplicating data
        num_batches = num_batches - (num_batches % self.num_replicas)

        # return number of batches for each GPU
        # could do this and the above all in one line using // division but keep separate to show
        # that even if strategy might be different, we still need to return batches per replica
        num_batches = int(num_batches / self.num_replicas)

        return num_batches

    def set_epoch(self, epoch: int) -> None:
        # print("[RANK {}] PDBSamplerDistributed (shuffle={}) epoch is being set to {}".format(
        #     os.getenv("LOCAL_RANK", '0'), self.shuffle, epoch))
        self.epoch = epoch
