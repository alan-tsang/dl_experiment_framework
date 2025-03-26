import types
import warnings
from typing import Union, Optional, Callable, List, Dict
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from fontTools.misc.iterTools import batched
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(
        self,
        data_source: Union[str, Dataset, DatasetDict],
        process_fn: Optional[Callable[[Dict], Dict]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
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
        self.metadata = metadata or {"description": "Custom Dataset"}
        self.data_format = data_format
        self.dataset = None


    @abstractmethod
    def _set_dataset(self, *args, **kwargs):
        pass

    @classmethod
    def get_batch_loader(
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
        """获取PyTorch数据加载器"""
        formatted_dataset = dataset.with_format("torch")
        return DataLoader(
            formatted_dataset,  # 转换为PyTorch Dataset
            batch_size = batch_size,
            pin_memory = pin_memory,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            shuffle = shuffle,
            **loader_kwargs
        )

    def _prepare_data(self):
        if isinstance(self.dataset, Dataset):
            assert isinstance(self.process_fn, dict),\
                ("process_fn is a dict, but dataset is a single dataset.\
                  process_fn should be a function or None.")
            assert isinstance(self.filter_fn, dict),\
                ("filter_fn is a dict, but dataset is a single dataset.\
                  filter_fn should be a function or None.")
            self.dataset = (self.dataset.filter(self.filter_fn)\
                            .map(self.process_fn, batched=True, batch_size = 1024, ))
        elif isinstance(self.dataset, DatasetDict):
            # 如果process_fn和filter_fn是函数，转换为字典
            if isinstance(self.process_fn, types.FunctionType):
                warnings.warn("process_fn is a function, but dataset is a DatasetDict.\
                               process_fn will be applied to all splits.")
                self.process_fn = {k: self.process_fn for k in self.dataset.keys()}
            if isinstance(self.filter_fn, types.FunctionType):
                self.filter_fn = {k: self.filter_fn for k in self.dataset.keys()}
                warnings.warn("filter_fn is a function, but dataset is a DatasetDict. filter_fn will be applied to all splits.")
            for split in self.dataset.keys():
                assert split in self.process_fn,\
                    (f"process_fn is missing {split} split.")
                assert split in self.filter_fn,\
                    (f"filter_fn is missing {split} split.")
                self.dataset[split] = (self.dataset[split].filter(self.filter_fn[split])\
                                       .map(self.process_fn[split], batched=True, batch_size = 1024))



    def save_to_disk(self, path: str):
        """保存到本地"""
        self.dataset.save_to_disk(path)

    @classmethod
    def from_cfg(cls, path: str, **kwargs) -> "BaseDataset":
        """从本地加载"""
        return cls(data_source = path, **kwargs)

    def push_to_hub(self, repo_id: str, **kwargs):
        """上传到HuggingFace Hub"""
        self.dataset.push_to_hub(repo_id, **kwargs)
