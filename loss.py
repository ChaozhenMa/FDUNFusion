import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image

from config import get_loss_config, get_path_config, get_model_config
from model import FracGradCalculator, GradientCalculator


class WeightConstraintLoss(nn.Module):
    def __init__(self, alpha=0.1, save_dir="weight_debug", save_epochs=None, frac_order=0.1):
        super(WeightConstraintLoss, self).__init__()
        self.alpha = alpha
        self.save_dir = save_dir
        self.frac_calc = None
        self.frac_order = frac_order
        self.diff_matrix = None  # 延迟初始化的显存缓存

        self.save_epochs = save_epochs if save_epochs is not None else list(range(0, 201, 20))
        self.saved_epochs = set()
        os.makedirs(save_dir, exist_ok=True)

    def forward(self, predicted_weights, U, V, current_epoch=None):
        W1_pred, W2_pred = predicted_weights[:, 0:1], predicted_weights[:, 1:2]

        def LC_vectorized(img):
            if img.shape[1] == 3:
                gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
            else:
                gray = img

            gray_uint8 = (gray * 255).clamp(0, 255).long()
            batch_size, _, height, width = gray_uint8.shape

            # 使用缓存的 diff_matrix 极大地节省显存开销
            if self.diff_matrix is None or self.diff_matrix.device != img.device:
                i_arr = torch.arange(256, device=img.device)
                j_arr = torch.arange(256, device=img.device)
                self.diff_matrix = torch.abs(i_arr.unsqueeze(1) - j_arr.unsqueeze(0)).float()

            lc_weights = torch.zeros((batch_size, 1, height, width), device=img.device, dtype=torch.float32)

            for b in range(batch_size):
                img_flat = gray_uint8[b, 0].flatten()
                hist = torch.bincount(img_flat, minlength=256).float()
                P = hist / (height * width)
                Sal_Tab = torch.sum(P.unsqueeze(1) * self.diff_matrix, dim=0)
                lc_weights[b, 0] = Sal_Tab[img_flat].reshape(height, width)

            return lc_weights

        def WviaLC2(U, V):
            if U.shape[1] == 3:
                U_gray = 0.299 * U[:, 0:1] + 0.587 * U[:, 1:2] + 0.114 * U[:, 2:3]
                V_gray = 0.299 * V[:, 0:1] + 0.587 * V[:, 1:2] + 0.114 * V[:, 2:3]
            else:
                U_gray, V_gray = U, V

            L_U = LC_vectorized(U_gray)
            L_V = LC_vectorized(V_gray)

            def k_produce(img, ML, MU):
                img_double = img.type(torch.float32) / 255.0 if img.dtype == torch.uint8 else img.type(torch.float32)
                return (MU - ML) * img_double / 255 + ML

            Cir = k_produce(U_gray * 255, 10, 70)
            Cvi = k_produce(V_gray * 255, 10, 70)
            eps = 1e-8

            x_log = torch.log(L_U + eps) + Cir * torch.log(U_gray + eps)
            y_log = torch.log(L_V + eps) + Cvi * torch.log(V_gray + eps)

            max_log = torch.max(x_log, y_log)
            x_exp = torch.exp(x_log - max_log)
            y_exp = torch.exp(y_log - max_log)

            W_U_gray = x_exp / (x_exp + y_exp + eps)
            W_V_gray = 1 - W_U_gray
            return W_U_gray, W_V_gray

        W1_ref, W2_ref = WviaLC2(U, V)

        if (current_epoch is not None and current_epoch in self.save_epochs and current_epoch not in self.saved_epochs):
            if U.shape[0] > 0:
                self.save_weight_image(W1_ref[0, 0], f"epoch_{current_epoch:03d}_W1_ref.png")
                self.save_weight_image(W2_ref[0, 0], f"epoch_{current_epoch:03d}_W2_ref.png")
                self.save_weight_image(W1_pred[0, 0], f"epoch_{current_epoch:03d}_W1_pred.png")
                self.save_weight_image(W2_pred[0, 0], f"epoch_{current_epoch:03d}_W2_pred.png")
                print(f"✅ 保存 epoch_{current_epoch:03d} 关键权重图像")
                self.saved_epochs.add(current_epoch)

        weight_loss = F.mse_loss(W1_pred, W1_ref) + F.mse_loss(W2_pred, W2_ref)
        return weight_loss * self.alpha

    def save_weight_image(self, weight_tensor, filename):
        weight_np = weight_tensor.detach().cpu().numpy()
        weight_np = (weight_np - weight_np.min()) / (weight_np.max() - weight_np.min() + 1e-8)
        weight_uint8 = (weight_np * 255).astype(np.uint8)
        filepath = os.path.join(self.save_dir, filename)
        Image.fromarray(weight_uint8).save(filepath)


class CombinedFusionLoss(nn.Module):
    def __init__(self, use_fractional=None, frac_order=None, **kwargs):
        config = get_loss_config(**kwargs)
        path_config = get_path_config()
        model_config = get_model_config()

        super(CombinedFusionLoss, self).__init__()
        self.alpha = config['alpha']
        self.gamma = config['gamma']
        self.mu = config['mu']
        self.weight_alpha = config['weight_alpha']
        self.save_epochs = config.get('save_epochs', None)

        self.use_fractional = model_config['use_fractional']
        self.frac_order = model_config['order']
        self.gradient_calc = None

        self.weight_constraint = WeightConstraintLoss(
            alpha=self.weight_alpha,
            save_dir=path_config['debug_weights'],
            save_epochs=self.save_epochs,
            frac_order=self.frac_order
        )

    def _init_gradient_calc(self, device):
        if self.gradient_calc is None:
            if self.use_fractional:
                self.gradient_calc = FracGradCalculator(order=self.frac_order, device=device)
            else:
                self.gradient_calc = GradientCalculator(device=device)

    def forward(self, fused, U, V, weights, current_epoch=None):
        W1, W2 = weights[:, 0:1], weights[:, 1:2]
        self._init_gradient_calc(fused.device)

        intensity_loss = (W1 * (fused - V) ** 2 + W2 * (fused - U) ** 2).mean()

        _, _, grad_fused = self.gradient_calc.compute_gradient(fused)
        _, _, grad_U = self.gradient_calc.compute_gradient(U)
        _, _, grad_V = self.gradient_calc.compute_gradient(V)

        max_grad = torch.max(grad_U, grad_V)
        gradient_loss = F.l1_loss(grad_fused, max_grad)
        brightness_loss = -torch.mean(fused ** 2)

        original_loss = self.alpha * intensity_loss + self.gamma * gradient_loss + self.mu * brightness_loss
        weight_constraint_loss = self.weight_constraint(weights, U, V, current_epoch=current_epoch)

        total_loss = original_loss + weight_constraint_loss
        return total_loss, intensity_loss, gradient_loss, brightness_loss, weight_constraint_loss