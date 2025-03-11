import os
from typing import Union, Optional, Callable, List, Dict
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from .base_dataset import BaseDataset
from ..common.registry import registry

@registry.register_dataset("BaseMapDataset")
class BaseMapDataset(BaseDataset):
    def __init__(
        self,
        data_source: Union[str, Dataset, DatasetDict],
        process_fn: Optional[Callable[[Dict], Dict]] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        metadata: Optional[Dict] = None,
        data_format: Optional[str] = None,
        split: str = "train",
        split_ratios: Optional[tuple] = (0.8, 0.1, 0.1),
        *args,
        **kwargs
    ):
        super().__init__(
            data_source,
            process_fn,
            filter_fn,
            metadata,
            data_format
        )
        self.split = split
        self.split_ratios = split_ratios
        self._set_dataset(data_source, split = split)

        # 自动数据集划分
        if split_ratios and isinstance(self.dataset, Dataset):
            self.auto_split(split_ratios)

        self._prepare_data()


    def _set_dataset(self, data_source, split, data_format=None):
        # 加载数据集
        if isinstance(data_source, str):
            if os.path.exists(data_source):
                if data_format is None:
                    data_format = infer_data_format(data_source)
                    if data_format is None:
                        raise ValueError("无法推断数据格式，请通过data_format参数指定。")

                self.dataset = load_dataset(
                    data_format,
                    split = split,
                    data_files=data_source,
                )
            else:
                # HuggingFace Hub 名称
                self.dataset = load_dataset(data_source, streaming=True)

        elif isinstance(data_source, (Dataset, DatasetDict)):
            self.dataset = data_source[split] if isinstance(data_source, DatasetDict) else data_source
        else:
            raise ValueError("不支持的数据源类型")


    def auto_split(self, ratios: tuple):
        """
        不确定能不能支持流式数据集
        :param ratios:
        :return:
        """
        """自动划分训练/验证/测试集"""
        train_ratio, val_ratio, test_ratio = ratios
        total = sum(ratios)
        train_size = train_ratio / total

        train_temp = self.dataset.train_test_split(test_size = 1 - train_size)
        temp = train_temp["test"]

        # 第二次划分：验证集和测试集
        val_test = temp.train_test_split(test_size = test_ratio / (val_ratio + test_ratio))

        self.dataset = DatasetDict(
            {
                "train": train_temp["train"],
                "val": val_test["train"],
                "test": val_test["test"]
            }
        )[self.split]


    def get_subset(self, indices: List[int]) -> "BaseMapDataset":
        """获取子集"""
        return BaseMapDataset(
            data_source = self.dataset.select(indices),
            split = self.split,
            process_fn = self.process_fn,
            filter_fn = self.filter_fn,
            metadata = self.metadata
        )


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    @property
    def dataset_card(self) -> Dict:
        """生成数据集卡片"""
        return {
            **self.metadata,
            "splits": list(self.dataset.keys()) if isinstance(self.dataset, DatasetDict) else [self.split],
            "size": len(self),
            "streaming": False,
            "features": self.dataset.features if hasattr(self.dataset, 'features') else None
        }


@registry.register_dataset("BaseIterableDataset")
class BaseIterableDataset(BaseDataset):
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
        super().__init__(
            data_source,
            process_fn,
            filter_fn,
            metadata,
        )
        self._set_dataset(data_source, data_format)

        self._prepare_data()


    def _set_dataset(self, data_source: Union[str, Dataset, DatasetDict],
                     data_format: Optional[str] = None,):
        # 加载数据集
        if isinstance(data_source, str):
            if os.path.exists(data_source):
                if os.path.isdir(data_source):
                    raise ValueError("流式模式不支持加载Dataset目录，请使用原始数据文件。")

                if data_format is None:
                    data_format = infer_data_format(data_source)
                    if data_format is None:
                        raise ValueError("无法推断数据格式，请通过data_format参数指定。")

                self.dataset = load_dataset(
                    data_format,
                    # wtf?
                    split = 'train',
                    data_files=data_source,
                    streaming=True
                )
            else:
                # HuggingFace Hub 名称
                self.dataset = load_dataset(data_source, streaming=True)
        elif isinstance(data_source, (Dataset, DatasetDict)):
            self.dataset = data_source.to_iterable_dataset()
        elif isinstance(data_source, IterableDataset):
            self.dataset = data_source
        else:
            raise ValueError("不支持的数据源类型")


    def __iter__(self):
        yield from self.dataset

    @property
    def dataset_card(self) -> Dict:
        """生成数据集卡片"""
        return {
            **self.metadata,
            "streaming": True,
            "features": self.dataset.features if hasattr(self.dataset, 'features') else None
        }


def infer_data_format(path: str) -> Optional[str]:
    """根据文件扩展名推断数据格式"""
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    format_mapping = {
        'csv': 'csv',
        'json': 'json',
        'jsonl': 'json',
        'txt': 'text',
        'text': 'text',
        'tsv': 'csv'  # 可能需要指定分隔符
    }
    return format_mapping.get(ext)
