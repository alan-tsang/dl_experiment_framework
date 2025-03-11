from typing import Union, Optional, Callable, List, Dict
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
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


    def get_batch_loader(
            self,
            batch_size: int = 1,
            pin_memory: bool = False,
            sampler = None,
            num_workers: int = 0,
            collate_fn = None,
            **loader_kwargs
    ) -> DataLoader:
        """获取PyTorch数据加载器"""
        formatted_dataset = self.dataset.with_format("torch")
        return DataLoader(
            formatted_dataset,  # 转换为PyTorch Dataset
            batch_size = batch_size,
            pin_memory = pin_memory,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            shuffle = False,
            **loader_kwargs
        )

    def _prepare_data(self):
        """应用预处理和过滤"""
        self.dataset = self.dataset.map(self.process_fn).filter(self.filter_fn)

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
