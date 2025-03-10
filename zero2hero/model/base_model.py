"""
adapted from salesforce's lavis: https://github.com/salesforce/LAVIS/blob/main/lavis/models/base_model.py
"""
import inspect
import logging

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()


    @property
    def device(self):
        return list(self.parameters())[0].device


    def load_checkpoint(self, cached_path):
        """Load from a pretrained checkpoint.
        Maybe this should expect no mismatch in the model keys and the checkpoint keys.
        """

        checkpoint = torch.load(cached_path, map_location="cpu")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % cached_path)

        return msg


    @classmethod
    def from_cfg(cls, cfg):

        """根据配置文件动态构建模型，支持以下功能：
        - 自动匹配子类构造器（当继承BaseModel时）
        - 预训练权重加载
        - 设备分配

        example:
        >>> cfg = {
        ... "model_type": "TransformersToy",
        ... "layers": 6
        ... }
        >>> model = BaseModel.from_cfg(cfg)

        >>> cfg = {
        ... "layers": 6
        ... }
        >>> model = TransformersToy.from_cfg(cfg)

        """

        # 解析配置类型（支持OmegaConf/argparse/dict）
        if hasattr(cfg, 'get'):
            # OmegaConf等类dict对象
            config_dict = dict(cfg)
        else:
            config_dict = vars(cfg) if not isinstance(cfg, dict) else cfg

        # 动态获取子类构造器（允许继承时自动匹配）
        if 'model_type' in config_dict:
            # 多态 + 注册机制
            """
            model = BaseModel.from_cfg(cfg)
            """
            from ..common.registry import registry
            registry.get_model_class(config_dict['model_type'])
            model_cls = registry.get_model_class(config_dict['model_type'])

        elif cls != BaseModel:
            # 如果直接通过子类调用，使用子类自身
            """
            MyTransformerLM.from_cfg(cfg)
            """
            model_cls = cls
        else:
            raise ValueError("Must specify model_type when using BaseModel directly")

        # 提取模型构造参数（过滤非法参数）
        valid_args = inspect.signature(model_cls.__init__).parameters
        model_args = {k: v for k, v in config_dict.items() if k in valid_args}

        # 初始化模型实例
        model = model_cls(**model_args)

        # 加载预训练权重(可选)
        if 'pretrained' in config_dict:
            load_result = model.load_checkpoint(config_dict['pretrained'])
            # 检查权重加载（根据需求调整）
            if len(load_result.missing_keys) > 0 and not config_dict.get('allow_missing_keys', False):
                import warnings
                warnings.warn(f"Missing keys in checkpoint: {load_result.missing_keys}")

        # 设备分配（优先使用配置文件指定设备）
        device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, str):
            device = torch.device(device)
        model = model.to(device)

        return model
