import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

# 配置参数
config = {
    "dataset_size": 1024,  # 总数据量
    "input_channels": 3,  # 输入通道数
    "image_size": 32,  # 图像尺寸
    "batch_size": 64,  # 批大小
    "epochs": 5,  # 训练轮次
    "lr": 1e-3,  # 学习率
    "world_size": 1,  # 并行进程数（需<=可用GPU数）
}


class DynamicDataset(Dataset):
    def __init__(self, original_data):
        self.original_data = original_data.clone()
        self.replaced_data = original_data.clone()
        self.replaced_metrics = torch.full(
            (len(original_data), ),
            fill_value = float('-inf'),
             device=original_data.device
        )
        self.global_indices = torch.arange(len(original_data))

    def __getitem__(self, index):
        global_idx = self.global_indices[index].item()
        if self.replaced_metrics[global_idx] > float('-inf'):
            return self.replaced_data[global_idx], global_idx
        return self.original_data[global_idx], global_idx

    def __len__(self):
        return len(self.original_data)

    def update_samples(self, global_indices, new_data, metrics):
        device = self.replaced_metrics.device
        global_indices = global_indices.to(device)
        mask = metrics.to(device) > self.replaced_metrics[global_indices]

        valid_indices = global_indices[mask]
        valid_data = new_data[mask].to(device)
        valid_metrics = metrics[mask].to(device)

        self.replaced_data[valid_indices] = valid_data
        self.replaced_metrics[valid_indices] = valid_metrics


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 2, padding = 1),  # 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride = 2, padding = 1),  # 32x8x8
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride = 2, padding = 1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def sync_replacements(rank, world_size, global_indices, local_data, local_metrics):
    """分布式同步替换候选"""
    # 收集所有节点的数据
    gathered_indices = [torch.zeros_like(global_indices) for _ in range(world_size)]
    gathered_data = [torch.zeros_like(local_data) for _ in range(world_size)]
    gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(world_size)]

    dist.all_gather(gathered_indices, global_indices)
    dist.all_gather(gathered_data, local_data)
    dist.all_gather(gathered_metrics, local_metrics)

    # 合并数据
    all_indices = torch.cat(gathered_indices)
    all_data = torch.cat(gathered_data)
    all_metrics = torch.cat(gathered_metrics)

    # 筛选每个索引的最佳候选
    unique_indices, inverse_idx = torch.unique(all_indices, return_inverse = True)
    best_data = []
    best_metrics = []

    for idx in unique_indices:
        mask = (all_indices == idx)
        best_pos = torch.argmax(all_metrics[mask])
        best_data.append(all_data[mask][best_pos])
        best_metrics.append(all_metrics[mask][best_pos])

    return unique_indices, torch.stack(best_data), torch.stack(best_metrics)


def train(rank, world_size, config):
    # 初始化分布式训练
    dist.init_process_group(
        backend = 'nccl',
        init_method = 'tcp://127.0.0.1:12345',
        rank = rank,
        world_size = world_size
    )
    torch.cuda.set_device(rank)

    # 生成示例数据（所有进程使用相同初始数据）
    torch.manual_seed(0)
    dummy_data = torch.randn(
        config["dataset_size"],
        config["input_channels"],
        config["image_size"],
        config["image_size"]
    )
    dataset = DynamicDataset(dummy_data)

    # 准备数据加载器
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas = world_size,
        rank = rank
    )
    dataloader = DataLoader(
        dataset,
        batch_size = config["batch_size"],
        sampler = sampler,
        pin_memory = True
    )

    # 初始化模型
    model = AutoEncoder().cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids = [rank])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"])

    # 训练循环
    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)
        model.train()

        for x, global_indices in dataloader:
            x = x.cuda(non_blocking = True)
            global_indices = global_indices.cuda(non_blocking = True)

            # 前向传播
            output = model(x)
            loss = criterion(output, x)

            # 计算替换指标
            with torch.no_grad():
                mse = torch.mean((output - x) ** 2, dim = [1, 2, 3])
                new_mse = mse * 0.8  # 假设模型输出质量提升20%

                replace_mask = new_mse < mse
                selected_global = global_indices[replace_mask]
                replace_data = output[replace_mask]
                replace_metrics = -new_mse[replace_mask]  # 最大化指标

            # 分布式同步
            if selected_global.numel() > 0:
                sync_indices, sync_data, sync_metrics = sync_replacements(
                    rank,
                    world_size,
                    selected_global,
                    replace_data,
                    replace_metrics
                )

                # 更新数据集
                dataset.update_samples(
                    sync_indices.cuda(),
                    sync_data.cuda(),
                    sync_metrics.cuda()
                )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印日志（仅主进程）
            if rank == 0:
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} | "
                    f"Replaced {len(sync_indices)} samples"
                )


def main():
    # 生成示例数据
    print("Generating dummy data...")

    # 启动分布式训练
    mp.spawn(
        train,
        args = (config["world_size"], config),
        nprocs = config["world_size"],
        join = True
    )


if __name__ == "__main__":
    main()