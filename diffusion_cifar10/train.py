"""
CIFAR10 Diffusion Model 训练脚本
支持 UNet 和 DiT 两种架构
使用 PyTorch DDP 进行 8 GPU 分布式训练

用法:
    torchrun --nproc_per_node=8 train.py --arch unet
    torchrun --nproc_per_node=8 train.py --arch dit
    torchrun --nproc_per_node=8 train.py --arch unet_improved
"""
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from pathlib import Path

from model import UNet, UNetImproved
from dit_model import DiT
from diffusion import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Model on CIFAR10')
    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'unet_improved', 'dit'],
                        help='模型架构: unet / unet_improved / dit')
    parser.add_argument('--epochs', type=int, default=500,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='每张GPU的batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='扩散步数')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Beta schedule')
    parser.add_argument('--loss_type', type=str, default='l2',
                        choices=['l1', 'l2', 'huber'],
                        help='损失函数类型')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA衰减率')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='每隔多少epoch保存一次')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='每隔多少epoch采样一次')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--use_ddim', action='store_true',
                        help='采样时使用DDIM加速')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM采样步数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    # DiT 参数
    parser.add_argument('--dit_depth', type=int, default=12,
                        help='DiT Transformer层数')
    parser.add_argument('--dit_hidden_size', type=int, default=384,
                        help='DiT 隐藏维度')
    parser.add_argument('--dit_num_heads', type=int, default=6,
                        help='DiT 注意力头数')
    parser.add_argument('--dit_patch_size', type=int, default=2,
                        help='DiT patch大小')
    return parser.parse_args()


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.model = model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def setup_distributed():
    """初始化分布式训练"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def create_model(args, device):
    """根据参数创建模型"""
    if args.arch == 'unet':
        model = UNet(c_in=3, c_out=3, time_dim=256)
    elif args.arch == 'unet_improved':
        model = UNetImproved(
            in_channels=3, model_channels=128, out_channels=3,
            num_res_blocks=2, attention_resolutions=(16, 8),
            channel_mult=(1, 2, 2, 2), time_embed_dim=512
        )
    elif args.arch == 'dit':
        model = DiT(
            img_size=32,
            patch_size=args.dit_patch_size,
            in_channels=3,
            hidden_size=args.dit_hidden_size,
            depth=args.dit_depth,
            num_heads=args.dit_num_heads,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = model.to(device)
    return model


def get_dataloader(args, rank, world_size):
    """创建CIFAR10数据加载器"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler


@torch.no_grad()
def sample_and_save(model, diffusion, epoch, args, device, rank):
    """生成并保存样本图像"""
    if rank != 0:
        return

    model.eval()
    n_samples = 64

    samples = diffusion.sample(
        model, num_samples=n_samples, img_size=32, channels=3,
        use_ddim=args.use_ddim, ddim_steps=args.ddim_steps, device=device
    )

    # 从 [-1, 1] 转换到 [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    save_path = Path(args.output_dir) / 'samples'
    save_path.mkdir(parents=True, exist_ok=True)

    save_image(samples, save_path / f'epoch_{epoch:04d}.png', nrow=8)
    print(f"[Epoch {epoch}] 采样图像已保存到 {save_path / f'epoch_{epoch:04d}.png'}")


def train(args):
    """主训练函数"""
    # 初始化分布式
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        print(f"========================================")
        print(f"  CIFAR10 Diffusion Model Training")
        print(f"  Architecture: {args.arch}")
        print(f"  GPUs: {world_size}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Timesteps: {args.num_timesteps}")
        print(f"  Beta schedule: {args.beta_schedule}")
        print(f"  Loss type: {args.loss_type}")
        print(f"  EMA decay: {args.ema_decay}")
        print(f"========================================")

    # 创建输出目录
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / 'samples').mkdir(parents=True, exist_ok=True)

    # 创建模型
    model = create_model(args, device)
    model_ddp = DDP(model, device_ids=[local_rank])

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"模型参数量: {num_params:.2f}M")

    # 创建EMA
    ema = EMA(model, decay=args.ema_decay)

    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # 优化器
    optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=args.lr, weight_decay=0.01)

    # 学习率调度器 (warmup + cosine decay)
    warmup_steps = 5000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.epochs * 390 - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 数据加载
    dataloader, sampler = get_dataloader(args, rank, world_size)

    # 恢复训练
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        ema.load_state_dict(checkpoint['ema'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        if rank == 0:
            print(f"从 epoch {start_epoch} 恢复训练")

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model_ddp.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            # 随机采样时间步
            t = torch.randint(0, args.num_timesteps, (batch_size,), device=device)

            # 前向传播 + 计算损失 (混合精度)
            with torch.amp.autocast('cuda'):
                loss = diffusion.compute_loss(model_ddp, images, t, loss_type=args.loss_type)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model_ddp.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 更新EMA
            ema.update()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0 and batch_idx % 100 == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Batch {batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Avg Loss: {avg_loss:.4f} "
                    f"LR: {lr:.6f}"
                )

        # Epoch 结束
        avg_epoch_loss = epoch_loss / num_batches
        if rank == 0:
            print(f"[Epoch {epoch}] 平均损失: {avg_epoch_loss:.4f}")

        # 采样 (使用EMA参数)
        if (epoch + 1) % args.sample_interval == 0:
            ema.apply_shadow()
            sample_and_save(model, diffusion, epoch, args, device, rank)
            ema.restore()

        # 保存checkpoint
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            ckpt_path = Path(args.output_dir) / 'checkpoints' / f'ckpt_epoch_{epoch:04d}.pt'
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.state_dict(),
                'args': vars(args),
            }, ckpt_path)
            print(f"Checkpoint 已保存: {ckpt_path}")

    # 保存最终模型
    if rank == 0:
        ema.apply_shadow()
        final_path = Path(args.output_dir) / 'checkpoints' / 'final_model.pt'
        torch.save({
            'model': model.state_dict(),
            'args': vars(args),
        }, final_path)
        print(f"最终模型已保存: {final_path}")

    cleanup_distributed()


if __name__ == '__main__':
    args = parse_args()
    train(args)
