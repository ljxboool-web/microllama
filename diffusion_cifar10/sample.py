"""
从训练好的模型生成图像

用法:
    python sample.py --ckpt output/checkpoints/final_model.pt --arch unet --num_samples 64
    python sample.py --ckpt output/checkpoints/final_model.pt --arch dit --use_ddim --ddim_steps 50
"""
import argparse
import torch
from torchvision.utils import save_image
from pathlib import Path

from model import UNet, UNetImproved
from dit_model import DiT
from diffusion import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description='Sample from trained Diffusion Model')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint路径')
    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'unet_improved', 'dit'])
    parser.add_argument('--num_samples', type=int, default=64, help='生成图像数量')
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine')
    parser.add_argument('--use_ddim', action='store_true', help='使用DDIM加速采样')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--output', type=str, default='generated.png')
    parser.add_argument('--device', type=str, default='cuda')
    # DiT 参数
    parser.add_argument('--dit_depth', type=int, default=12)
    parser.add_argument('--dit_hidden_size', type=int, default=384)
    parser.add_argument('--dit_num_heads', type=int, default=6)
    parser.add_argument('--dit_patch_size', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 加载模型
    print(f"加载模型: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)

    # 如果checkpoint中保存了参数，则使用
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        args.arch = saved_args.get('arch', args.arch)
        args.num_timesteps = saved_args.get('num_timesteps', args.num_timesteps)
        args.beta_schedule = saved_args.get('beta_schedule', args.beta_schedule)
        if args.arch == 'dit':
            args.dit_depth = saved_args.get('dit_depth', args.dit_depth)
            args.dit_hidden_size = saved_args.get('dit_hidden_size', args.dit_hidden_size)
            args.dit_num_heads = saved_args.get('dit_num_heads', args.dit_num_heads)
            args.dit_patch_size = saved_args.get('dit_patch_size', args.dit_patch_size)

    # 创建模型
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
            img_size=32, patch_size=args.dit_patch_size, in_channels=3,
            hidden_size=args.dit_hidden_size, depth=args.dit_depth,
            num_heads=args.dit_num_heads,
        )

    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"架构: {args.arch}, 参数量: {num_params:.2f}M")

    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # 生成图像
    sampling_method = "DDIM" if args.use_ddim else "DDPM"
    steps = args.ddim_steps if args.use_ddim else args.num_timesteps
    print(f"使用 {sampling_method} 采样 {args.num_samples} 张图像 (steps={steps})...")

    samples = diffusion.sample(
        model,
        num_samples=args.num_samples,
        img_size=32,
        channels=3,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
        device=device,
    )

    # 保存
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    nrow = int(args.num_samples ** 0.5)
    save_image(samples, args.output, nrow=nrow)
    print(f"生成图像已保存到: {args.output}")


if __name__ == '__main__':
    main()
