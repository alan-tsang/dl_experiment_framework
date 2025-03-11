import argparse
import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from torch import Tensor

from ..common.registry import registry
from ..common.util import first_call_warning
from ..dist import (broadcast_object_list, collect_results,
                           is_main_process)


class BaseMetric(metaclass=ABCMeta):
    """Base class for a metric.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseMetric` should assign a meaningful value to the
    class attribute `default_prefix`. See the argument `prefix` for details.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:

        if collect_dir is not None and collect_device != 'cpu':
            raise ValueError('`collec_dir` could only be configured when '
                             "`collect_device='cpu'`")

        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

        if self.prefix is None:
            print(
                'The prefix is not set in metric class '
                f'{self.__class__.__name__}.'
            )

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_batch: Any, data_samples: argparse.Namespace) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.'
            )

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]


@registry.register_metric("DumpResults")
class DumpResults(BaseMetric):
    """Dump model predictions to a pickle file for offline evaluation.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    def __init__(self,
                 out_file_path: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        if not out_file_path.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')
        self.out_file_path = out_file_path

    def process(self, data_batch: Any, predictions: argparse.Namespace) -> None:
        """Transfer tensors in predictions to CPU."""
        pred = predictions.logit.argmax(-1)
        ground_truth = data_batch['y_shift']
        self.results.extend((_to_cpu(pred), _to_cpu(ground_truth)))


    def compute_metrics(self, results: list) -> dict:
        """Dump the prediction results to a pickle file."""
        with open(self.out_file_path, 'wb') as f:
            pickle.dump(results, f)
        print(
            f'Results has been saved to {self.out_file_path}.')
        return {}

    @staticmethod
    def filter_metric(metrics: argparse.Namespace, mode: str):
        should_monitor = list(registry.get(f"cfg.training.{mode}_monitor").keys())
        monitored = list(vars(metrics).keys())
        if not all([m in monitored for m in should_monitor]):
            first_call_warning(
                f"filter_metric_{mode}",
                f"{mode} 时期，compute_metric 返回的Namespace中缺少监控的keys;"
                f"应该包含：{should_monitor}，实际包含：{monitored}"
            )
        filtered_metrics = argparse.Namespace()
        for key, value in vars(metrics).items():
            if key in should_monitor:
                setattr(filtered_metrics, key, value)
        return filtered_metrics

    # 要求metric里的值支持sum
    @staticmethod
    def average_batch_metric(metrics: List[argparse.Namespace]) -> argparse.Namespace:
        """
        average metrics in different data loader' s batches
        """
        if not metrics:
            return argparse.Namespace()
        average_metric = argparse.Namespace()
        for key in vars(metrics[0]).keys():
            values = [getattr(metric, key) for metric in metrics]
            average = sum(values) / len(values)
            setattr(average_metric, key, average)
        return average_metric


    @staticmethod
    def register_metrics_values(metrics: argparse.Namespace):
        for key, value in vars(metrics).items():
            registry.register(f"metric.{key}", value)



def _to_cpu(data: Any) -> Any:
    """Transfer all tensors to cpu."""
    if isinstance(data, Tensor):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data