import math
import os
import glob
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from contextlib import nullcontext
import torch.amp
from transformers import PreTrainedModel
import logging
import swanlab
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from src.models.ostrich.ostrich_models import OstrichModel
from src.models.ostrich.ostrich_configuration import OstrichModelConfig
from src.datasets.pretrain_datasets import PretrainDataset


# 计算学习率， 支持wormkup + 余弦模拟退火
# 0-wormup阶段 线性增长 min_lr-> lr
# wormup-total_step阶段
def get_lr(
    cur_step: int, total_step: int, min_lr: float, lr: float, warmup_rate: float
):
    warmup_step = int(total_step * warmup_rate)
    if cur_step <= warmup_step:
        return cur_step / warmup_step * lr
    elif cur_step > total_step:
        return min_lr
    else:
        return min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                ((cur_step - warmup_step) / (total_step - warmup_step)) * math.pi
            )
        )


def cleanup_old_checkpoints(save_dir: str, max_checkpoints: int = 3):
    """
    清理旧的checkpoint文件，只保留最新的max_checkpoints个文件
    """
    checkpoint_pattern = os.path.join(save_dir, "pretrain_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= max_checkpoints:
        return

    # 根据文件修改时间排序，最新的在前
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # 删除多余的checkpoint文件
    files_to_delete = checkpoint_files[max_checkpoints:]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"删除旧checkpoint: {file_path}")
        except OSError as e:
            print(f"删除checkpoint失败 {file_path}: {e}")


def train_epoch(
    epcoh: int,
    iter_per_epoch: int,
    dataloader: DataLoader,
    device,
    optimize: Optimizer,
    min_lr,
    lr,
    accumulation_steps,
    warmup_rate,
    model_dtype,
    clip_norm,
    model: PreTrainedModel,
    log_interval: int = 10,
    save_interval: int = 100,
    save_dir: str = "./resources/models/ostrich1",
    use_swanlab: bool = True,
    local_rank: int = 0,
):
    total_step = epcoh * iter_per_epoch
    amp_dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float16
    ctx = torch.autocast(device_type=("cpu" if device == "cpu" else "cuda"), dtype=amp_dtype)
    use_scaler = (model_dtype == "float16")
    scaler = torch.GradScaler(device=("cpu" if device == "cpu" else "cuda"), enabled=use_scaler)
    for epoch_index in range(epcoh):
        for idx, (X, Y, loss_mask) in enumerate(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)

            # 获取lr
            current_step = epoch_index * iter_per_epoch + idx
            current_learn_rate = get_lr(
                current_step, total_step, min_lr, lr, warmup_rate=warmup_rate
            )

            # 更新optimizer 的learning_rate
            for param_group in optimize.param_groups:
                param_group["lr"] = current_learn_rate

            # 使用混合精度训练

            with ctx:
                # 前向传播
                output = model(X, Y)
                loss = output["loss"] / accumulation_steps

                # 将loss mask展平
                loss_mask = loss_mask.view(-1)

                loss = torch.sum(loss * loss_mask) / loss_mask.sum()

            # 使用scaler进行混合精度的反向传播 计算梯度
            scaler.scale(loss).backward()

            # 梯度累计
            if (idx + 1) % accumulation_steps == 0:
                # 先取消梯度缩放，为了后面的梯度裁剪做准备，不设置，后续梯度裁剪无效
                scaler.unscale_(optimizer=optimize)

                # 进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

                # 执行优化器步骤
                scaler.step(optimizer=optimize)

                # 更新 scaler缩放因子
                scaler.update()

                # 清空梯度， 设置set_to_none 可以降低内存，防止下一次再次创建
                optimize.zero_grad(set_to_none=True)

            # 日志记录
            if (idx + 1) % log_interval == 0 and local_rank == 0:
                print(
                    f"Epoch:[{epoch_index + 1}/{epcoh}]({idx + 1}/{iter_per_epoch}) loss:{loss.item() * accumulation_steps} lr:{current_learn_rate}"
                )

            if use_swanlab and local_rank == 0:
                swanlab.log(
                    {
                        "loss": loss.item() * accumulation_steps,
                        "lr": optimize.param_groups[-1]["lr"],
                    }
                )

            # 模型保存
            if (idx + 1) % save_interval == 0:
                # 切换到评估模式进行保存，防止保存的模型自带梯度
                model.eval()

                # 确保保存目录存在
                os.makedirs(save_dir, exist_ok=True)

                ckp = f"""{save_dir}/pretrain_{idx}.pth"""
                if local_rank == 0:
                    state_dict = (
                        model.module.state_dict()
                        if isinstance(model, DDP)
                        else model.state_dict()
                    )
                    torch.save(state_dict, ckp)
                    cleanup_old_checkpoints(save_dir, max_checkpoints=3)

                model.train()


def init_model(config: OstrichModelConfig):
    """
    初始化 模型和分词器
    """

    from transformers import AutoTokenizer

    def count_trainable_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        if total >= 1e9:
            return f"{total/1e9:.3f} B"
        elif total >= 1e6:
            return f"{total/1e6:.3f} M"
        else:
            return str(total)

    tokenizer = AutoTokenizer.from_pretrained("./resources/models/ostrich1")
    model = OstrichModel(config=config)
    print(f"LLM 参数总量 {count_trainable_parameters(model)}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"Using DDP on rank {dist.get_rank()}/{dist.get_world_size()}")
    else:
        print("Running in single GPU mode")

    return model, tokenizer


def setup_ddp():
    """初始化DDP环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = setup_ddp()

    torch.manual_seed(42)
    llm_config = OstrichModelConfig()

    model, tokenzier = init_model(llm_config)

    # 设置训练参数，应该使用parser 来输出，但是我不想写
    """
    epcoh: int,
    iter_per_epoch: int,
    dataloader: DataLoader,
    device,
    optimize: Optimizer,
    min_lr,
    lr,
    accumulation_steps,
    warmup_rate,
    model_dtype,
    clip_norm,
    model: PreTrainedModel,
    log_interval: int = 10,
    save_interval: int = 100,
    save_dir: str = "./resource/models/ostrich1",
    use_swanlab: bool = False
    """
    epoch = 2
    device_type = "cuda"
    lr = 1e-4
    min_lr = lr / 10
    dataset = PretrainDataset(
        data_path="./resources/datasets/seq_monkey_datawhale_32k.jsonl",
        tokenizer=tokenzier,
    )

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
    )
    optimize = torch.optim.Adam(model.parameters(), lr=lr)
    accumulation_steps = 256
    iter_per_epoch = len(
        dataloader
    )  # 这个应该 按照 accumulation_steps 进行计算，但是我不想写，反正get_lr已经兼容了，试试看看
    warmup_rate = 0.05
    model_dtype = "bfloat16"
    clip_norm = 1.0
    use_swanlab = True

    if local_rank == 0:
        run = swanlab.init(
            project="Ostrich-LLM",
            experiment_name="ostrich-1.4B",
            config=llm_config.to_dict(),
        )

    try:
        train_epoch(
            epcoh=epoch,
            iter_per_epoch=iter_per_epoch,
            dataloader=dataloader,
            device=f"cuda:{local_rank}",
            optimize=optimize,
            min_lr=min_lr,
            lr=lr,
            accumulation_steps=accumulation_steps,
            warmup_rate=warmup_rate,
            model_dtype=model_dtype,
            clip_norm=clip_norm,
            model=model,
            use_swanlab=use_swanlab,
            save_interval=9000,
            local_rank=local_rank
        )
    finally:
        cleanup_ddp()
