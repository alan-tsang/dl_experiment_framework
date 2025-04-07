from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from zero2hero import *


class CustomDataset(Dataset):
    def __init__(self, vocab_n, max_len, data_n):
        self.vocab_n = vocab_n
        self.max_len = max_len
        self.data_n = data_n


    def __len__(self):
        # data_loader 根据__len__这个返回值来决定每个epoch的迭代次数
        return self.data_n


    def __getitem__(self, idx):
        # 生成单个样本，这里假设每个样本是 max_len 长度的序列
        x = torch.randint(3, self.vocab_n - 1, size=(self.max_len,))
        y = torch.add(x, 1)
        def regular(_):
            _[0] = 1
            _[-3:] = 0
            _[-2:] = 0
            _[-1] = 2
            return _

        x = regular(x)
        y = regular(y)

        return x, y


def collate_fn(batch):
    def make_mask(y, pad):
        def look_ahead_mask(size):
            attn_shape = (1, size, size)
            triu_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).to(torch.uint8)
            return triu_mask

        y_pad_mask = (y == pad).unsqueeze(-2)
        y_look_ahead_mask = look_ahead_mask(y.shape[-1])
        y_mask = y_pad_mask | y_look_ahead_mask
        return y_mask

    # multi batch -> list
    x, y = zip(*batch)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    y_input = y[:, :-1]
    y_shift = y[:, 1:]

    pad = 0
    x_mask = (x == pad).unsqueeze(-2)
    y_mask = make_mask(y_input, pad)

    return {
        'x': x,
        'y_input': y_input,
        'y_shift': y_shift,
        'x_mask': x_mask,
        'y_mask': y_mask
    }


from torch import nn
class DemoNet(BaseModel):
    def __init__(self, transformer, logit_generator):
        super().__init__()
        self.transformer = transformer
        self.logit_generator = logit_generator
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y_input, y_shift, x_mask, y_mask, *args, **kwargs):
        """
        if you dont want to refine the runner' s compute metric function,
        you should return loss and training monitors as a Namespace
        """
        out_prob = self.transformer(x, y_input, x_mask = x_mask, y_mask = y_mask)
        logit = self.logit_generator(out_prob)
        loss = self.criterion(logit.view(-1, logit.size(-1)), y_shift.view(-1))
        return dict(
            logit = logit,
            loss = loss
        )

    def generate(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MyMetric(BaseMetric):
    def __init__(self, collect_device = 'cpu', collect_dir = './tmp' ):
        super(MyMetric, self).__init__(collect_device, collect_dir = collect_dir)
        self.dataset_meta = 'bench_demo'

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        pred = data_samples["logit"].argmax(dim = -1)
        y_shift = data_batch['y_shift']
        self.results.append([pred, y_shift])


    def compute_metrics(self, results) -> dict:
        acc = []
        # 这里的result是每一个batch的结果
        for result in results:
            pred = result[0]
            y_shift = result[1]
            acc.append((pred == y_shift).float().mean().cpu())

        acc = (sum(acc) / len(acc)).item()

        return dict(valid_acc = acc)


class MyMetric2(BaseMetric):
    def __init__(self, collect_device = 'cpu', collect_dir = './tmp' ):
        super(MyMetric2, self).__init__(collect_device, collect_dir = collect_dir)
        self.dataset_meta = 'bench_demo'

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        pred = data_samples["classify_probs"]
        label = data_samples['label']
        self.results.append([pred, label])

    def compute_metrics(self, results) -> dict:
        acc = []
        # 这里的result实际上是每一个chunk的结果
        # result里是这个chunk的全部batch
        for result in results:
            pred = result[0]
            y_shift = result[1]
            # 取batch
            acc = [(x==y).mean() for x, y in zip(pred, y_shift)]

        acc = (sum(acc) / len(acc))

        return dict(test_acc = acc)


class DumpRunResult(DumpResults):

    def process(self, data_batch: Any, predictions: dict) -> None:
        logit = predictions['logit'].cpu().numpy()
        label = data_batch['y_shift'].cpu().numpy()
        classify_probs = logit.argmax(axis = -1)
        self.results.append({
            'logit': logit,
            'label': label,
            'classify_probs': classify_probs
        })


if __name__ == '__main__':
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument('--cfg', type=str, default='./default_cfg.yaml')
    args, _ = arg.parse_known_args()
    cfg_path = args.cfg
    cfg = load_cfg(cfg_path, from_cli = True)

    bench_dataset = CustomDataset(
        vocab_n = cfg.data.vocab_n,
        max_len = cfg.data.max_len,
        data_n = cfg.data.data_n
    )
    bench_loader = DataLoader(
        bench_dataset,
        batch_size = cfg.data.batch_size,
        shuffle = True,
        collate_fn = collate_fn
    )

    model_cfg = cfg.model
    model = registry.get_model_class(cfg.model.type)(**model_cfg)
    logit_generator = nn.Linear(model_cfg.d, model_cfg.vocab_n)
    net = DemoNet(model, logit_generator)
    # net.load_checkpoint('example/transformer_to_copy_str.pth')

    # optimizer = torch.optim.AdamW(net.parameters(), lr = cfg.optimizer.lr)
    evaluator = Evaluator([MyMetric(), DumpRunResult('./example/generated.pkl')])
    evaluator_test = Evaluator([MyMetric(), DumpRunResult('./example/test_result.pkl')])

    callbacks = [EarlyStopCallBack(monitor="loss")]

    runner = Runner(
        train_data_loader = bench_loader,
        valid_data_loader = bench_loader,
        test_data_loader = bench_loader,
        model = net,
        epochs = cfg.training.epochs,
        # optimizer = optimizer,
        runner_config = cfg,
        callbacks = callbacks,
        valid_evaluator = evaluator,
        test_evaluator = evaluator_test
    )
    runner.fit()
    runner.test()
    ############# test #############
    # 1. online: model load_model_checkpoint-> runner.test()
    # runner.test()
    # 2. offline: 现在的数据变换逻辑还是过于复杂了，容易出错
    data_samples = load('example/test_result_epoch_4.pkl')
    data_list = []
    for batch in bench_loader:
        data_list.append(batch)
    evaluator_offline = Evaluator([MyMetric2()])
    #
    print(evaluator_offline.offline_evaluate(data_samples, data_list, chunk_size = 10))
