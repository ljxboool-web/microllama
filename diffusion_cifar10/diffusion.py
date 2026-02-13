"""
DDPM (Denoising Diffusion Probabilistic Models) 扩散过程
"""
import torch
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion:
    """
    高斯扩散过程
    实现前向扩散 (加噪) 和反向去噪 (采样)
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 beta_schedule='linear', device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        # 定义 beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # 预计算扩散参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 前向扩散所需参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 反向去噪所需参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """Cosine beta schedule (improved DDPM)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _extract(self, a, t, x_shape):
        """提取对应时间步的参数"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散: q(x_t | x_0)
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, model, x_start, t, noise=None, loss_type='l2'):
        """
        计算训练损失
        L = E[||noise - model(x_t, t)||^2]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        单步反向去噪: p(x_{t-1} | x_t)
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # 预测均值
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device=None):
        """
        完整的反向去噪采样过程
        """
        if device is None:
            device = self.device

        b = shape[0]
        # 从纯噪声开始
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)

        return img

    @torch.no_grad()
    def ddim_sample(self, model, shape, ddim_steps=50, eta=0.0, device=None):
        """
        DDIM 加速采样
        """
        if device is None:
            device = self.device

        b = shape[0]
        # 均匀选择时间步子集
        step_size = self.num_timesteps // ddim_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device)
        timesteps = torch.flip(timesteps, [0])

        img = torch.randn(shape, device=device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            # 预测噪声
            predicted_noise = model(img, t_batch)

            # 获取参数
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # 预测 x_0
            pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # DDIM 更新
            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * predicted_noise
            noise = torch.randn_like(img) if i + 1 < len(timesteps) else 0

            img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise

        return img

    @torch.no_grad()
    def sample(self, model, num_samples, img_size=32, channels=3,
               use_ddim=False, ddim_steps=50, device=None):
        """
        生成样本的便捷接口
        """
        shape = (num_samples, channels, img_size, img_size)
        if use_ddim:
            return self.ddim_sample(model, shape, ddim_steps=ddim_steps, device=device)
        else:
            return self.p_sample_loop(model, shape, device=device)
