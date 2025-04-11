"""
BaseDataset: based on HuggingFace Datasets, with some additional features.
"""

import json
import os.path
import types
import warnings
from typing import Union, Optional, Callable, List, Dict
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(
        self,
        data_source: Union[str, Dataset, DatasetDict],
        process_fn: Optional[Callable[[Dict], Dict]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        process_first: bool = True,
        process_bs: int = 1,
        filter_bs: int = 1,
        metadata: Optional[Dict] = None,
        data_format: Optional[str] = None,
        *args,
        **kwargs
    ):
        """

        :param data_source: 数据路径/Hub名称/Dataset对象/DatasetDict对象
        :param process_fn:
        :param filter_fn:
        :param metadata:
        :param args:
        :param kwargs:
        """

        self.data_source = data_source
        self.process_fn = process_fn or (lambda x: x)
        self.filter_fn = filter_fn or (lambda x: True)
        self.process_first = process_first
        self.process_bs = process_bs
        self.filter_bs = filter_bs
        self.metadata = metadata or {}
        self.data_format = data_format
        self.dataset = None


    @abstractmethod
    def _set_dataset(self, *args, **kwargs):
        pass

    def get_batch_loader(self, split = None, *args, **kwargs):
        """

        :param split:
        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(self.dataset, DatasetDict):
            assert split is None, "split should be specified when dataset is a DatasetDict."
            dataset = self.dataset[split]
        elif isinstance(self.dataset, (Dataset, IterableDataset)):
            dataset = self.dataset
        else:
            raise ValueError("Unsupported dataset type.")

        return self.__class__._get_batch_loader(dataset, *args, **kwargs)

    @classmethod
    def _get_batch_loader(
            cls,
            dataset,
            batch_size: int = 1,
            pin_memory: bool = False,
            sampler = None,
            num_workers: int = 0,
            collate_fn = None,
            shuffle: bool = False,
            **loader_kwargs
    ) -> DataLoader:
        formatted_dataset = dataset.with_format("torch")
        return DataLoader(
            formatted_dataset,
            batch_size = batch_size,
            pin_memory = pin_memory,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            shuffle = shuffle,
            **loader_kwargs
        )


    def _prepare_data(self):
        def _map_filter(dataset, process_fn, filter_fn):
            if self.process_first:
                return dataset.map(process_fn, batched=True, batch_size = self.process_bs)\
                              .filter(filter_fn, batched=True, batch_size = self.filter_bs)
            else:
                return dataset.filter(filter_fn, batched=True, batch_size = self.filter_bs)\
                              .map(process_fn, batched=True, batch_size = self.process_bs)

        if isinstance(self.dataset, (Dataset, IterableDataset)):
            assert not isinstance(self.process_fn, dict), ("process_fn is a dict, but dataset is a single dataset. "
                                                           "process_fn should be a function or None.")
            assert not isinstance(self.filter_fn, dict),("filter_fn is a dict, but dataset is a single dataset."
                                                         "filter_fn should be a function or None.")
            self.dataset = _map_filter(self.dataset, self.process_fn, self.filter_fn)

        elif isinstance(self.dataset, DatasetDict):
            # 如果process_fn和filter_fn是函数，转换为字典
            if isinstance(self.process_fn, types.FunctionType):
                warnings.warn("process_fn is a function, but dataset is a DatasetDict. "
                              "process_fn will be applied to all splits.")
                self.process_fn = {k: self.process_fn for k in self.dataset.keys()}
            if isinstance(self.filter_fn, types.FunctionType):
                self.filter_fn = {k: self.filter_fn for k in self.dataset.keys()}
                warnings.warn("filter_fn is a function, but dataset is a DatasetDict."
                              " filter_fn will be applied to all splits.")

            for split in self.dataset.keys():
                process_fn = self.process_fn.get(split, None)
                filter_fn = self.filter_fn.get(split, None)
                self.dataset[split] = _map_filter(self.dataset[split], process_fn, filter_fn)


    def save_to_disk(self, path: str):
        os.makedirs(path, exist_ok=True)

        self.save_case(path)
        self.save_card(path)
        self.dataset.save_to_disk(path)


    def save_case(self, path: str):
        if isinstance(self.dataset, DatasetDict):
            for split in self.dataset.keys():
                case = str(self.dataset[split][0])
                with open(f"{path}/{split}_case.txt", "w") as f:
                    f.write(case)
        elif isinstance(self.dataset, Dataset):
            case = str(self.dataset[0])
            with open(f"{path}/dataset_case.txt", "w") as f:
                f.write(case)
        elif isinstance(self.dataset, IterableDataset):
            try:
                case = str(next(iter(self.dataset)))
            except StopIteration:
                warnings.warn("流式数据集未保存case data，因为现在已经没有数据，请检查数据源。")
            else:
                with open(f"{path}/dataset_case.txt", "w") as f:
                    f.write(case)


    def save_card(self, path: str):
        card = self.dataset_card
        with open(f"{path}/dataset_card.json", "w") as f:
            json.dump(card, f, indent=4)


    @property
    def dataset_card(self) -> Dict:
        return {
            **self.metadata,
        }


    @classmethod
    def from_cfg(cls, path: str, **kwargs) -> "BaseDataset":
        """兼容配置文件"""
        return cls(data_source = path, **kwargs)


    def push_to_hub(self, repo_id: str, **kwargs):
        """上传到HuggingFace Hub"""
        self.dataset.push_to_hub(repo_id, **kwargs)
