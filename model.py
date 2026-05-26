import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_model_config


class FracGradCalculator(nn.Module):
    def __init__(self, order=0.5, max_size=1024, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(FracGradCalculator, self).__init__()
        self.device = device
        self.order = order
        self.max_size = max_size
        self.matrix_cache = {}

    def _build_Ck_alpha_X(self, nx, alpha):
        Ck_x = torch.zeros(nx, nx, device=self.device)
        cka_x = torch.zeros(nx, device=self.device)
        cka_x[0] = 1
        for k in range(1, nx):
            cka_x[k] = cka_x[k - 1] * (1 - (alpha + 1) / k)
        ckaf_x = torch.flip(cka_x, [0])
        for i in range(nx):
            ckaf_CIRx = torch.roll(ckaf_x, shifts=-(i), dims=0)
            Ck_x[i, :nx - i] = ckaf_CIRx[:nx - i]
        return torch.rot90(Ck_x, k=1)

    def _build_Ck_alpha_Y(self, ny, alpha):
        Ck_y = torch.zeros(ny, ny, device=self.device)
        cka_y = torch.zeros(ny, device=self.device)
        cka_y[0] = 1
        for k in range(1, ny):
            cka_y[k] = cka_y[k - 1] * (1 - (alpha + 1) / k)
        ckaf_y = torch.flip(cka_y, [0])
        for i in range(ny):
            ckaf_CIRy = torch.roll(ckaf_y, shifts=-(i), dims=0)
            Ck_y[:ny - i, i] = ckaf_CIRy[:ny - i]
        return torch.fliplr(Ck_y)

    def compute_fractional_gradient(self, img):
        if img.dim() == 4:
            batch_size, channels, height, width = img.shape
            device = img.device

            cache_key_x = (height, self.order, device)
            cache_key_y = (width, self.order, device)

            if cache_key_x not in self.matrix_cache:
                self.matrix_cache[cache_key_x] = self._build_Ck_alpha_X(height, self.order).to(device)
            if cache_key_y not in self.matrix_cache:
                self.matrix_cache[cache_key_y] = self._build_Ck_alpha_Y(width, self.order).to(device)

            Ck_x = self.matrix_cache[cache_key_x]
            Ck_y = self.matrix_cache[cache_key_y]

            Dx_alpha_Z = torch.einsum('ij,bcjw->bciw', Ck_x, img)
            Dy_alpha_Z = torch.einsum('bcij,jw->bciw', img, Ck_y)

            magnitude = torch.sqrt(Dx_alpha_Z ** 2 + Dy_alpha_Z ** 2 + 1e-8)
            return Dx_alpha_Z, Dy_alpha_Z, magnitude
        else:
            raise ValueError("Input should be 4D tensor")

    def compute_gradient(self, img):
        return self.compute_fractional_gradient(img)


class GradientCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(
            device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(
            device)

    def compute_gradient(self, img):
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return gx, gy, magnitude


class FracProximalBlock(nn.Module):
    def __init__(self, gradient_calc):
        super(FracProximalBlock, self).__init__()
        self.gradient_calc = gradient_calc
        self.gradient_fusion_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, F, U_gx, U_gy, V_gx, V_gy, lambda_param, W1_pred=None, W2_pred=None, U=None, V=None):
        F_gx, F_gy, _ = self.gradient_calc.compute_gradient(F)
        max_gx, max_gy = torch.max(U_gx, V_gx), torch.max(U_gy, V_gy)
        grad_diff_x, grad_diff_y = F_gx - max_gx, F_gy - max_gy

        network_input = torch.cat([F, grad_diff_x, grad_diff_y], dim=1)
        F_refined = self.gradient_fusion_net(network_input)

        if W1_pred is not None and W2_pred is not None and U is not None and V is not None:
            attention_weights = torch.sigmoid(W1_pred * U + W2_pred * V)
            F_out = F + F_refined * lambda_param.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * attention_weights
        else:
            F_out = F + F_refined * lambda_param.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return F_out


class InfraredVisibleFusionDUN(nn.Module):
    def __init__(self, **kwargs):
        config = get_model_config(**kwargs)
        super(InfraredVisibleFusionDUN, self).__init__()

        self.num_stages = config['num_stages']
        self.fixed_weights = config['fixed_weights']
        self.use_fractional = config['use_fractional']

        if self.use_fractional:
            self.gradient_calc = FracGradCalculator(order=config['order'])
        else:
            self.gradient_calc = GradientCalculator()

        self.eta = nn.Parameter(torch.ones(self.num_stages) * 0.1)
        self.lambda_params = nn.Parameter(torch.ones(self.num_stages) * config['lambda_init'])
        self.beta_params = nn.Parameter(torch.ones(self.num_stages) * config['beta_init'])
        self.use_color_transfer = config['color_transfer']

        if not self.fixed_weights:
            self.weight_net = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 3, padding=1), nn.Softmax(dim=1)
            )

        self.proximal_blocks = nn.ModuleList([
            FracProximalBlock(gradient_calc=self.gradient_calc) for _ in range(self.num_stages)
        ])

    def color_transfer(self, grayscale_fused, color_reference):
        def rgb_to_yuv(rgb):
            r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.14713 * r - 0.28886 * g + 0.436 * b
            v = 0.615 * r - 0.51499 * g - 0.10001 * b
            return torch.cat([y, u, v], dim=1)

        def yuv_to_rgb(yuv):
            y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
            r = y + 1.13983 * v
            g = y - 0.39465 * u - 0.58060 * v
            b = y + 2.03211 * u
            return torch.cat([r, g, b], dim=1).clamp(0, 1)

        ref_yuv = rgb_to_yuv(color_reference)
        fused_yuv = torch.cat([grayscale_fused, ref_yuv[:, 1:3]], dim=1)
        return yuv_to_rgb(fused_yuv)

    def forward(self, U, V):
        batch_size, channels, height, width = U.shape
        original_V = V.clone() if channels == 3 else None

        if channels == 3:
            U_gray = 0.299 * U[:, 0:1] + 0.587 * U[:, 1:2] + 0.114 * U[:, 2:3]
            V_gray = 0.299 * V[:, 0:1] + 0.587 * V[:, 1:2] + 0.114 * V[:, 2:3]
        else:
            U_gray, V_gray = U, V

        self.intermediate_results = []
        self.intermediate_weights = []

        Fuse = (U_gray + V_gray) / 2
        self.intermediate_results.append(Fuse.clone())

        if self.fixed_weights:
            weights = torch.ones(batch_size, 2, height, width, device=U.device) * 0.5
            W1, W2 = weights[:, 0:1], weights[:, 1:2]
        else:
            weights = self.weight_net(torch.cat([U_gray, V_gray], dim=1))
            W1, W2 = weights[:, 0:1], weights[:, 1:2]

        self.intermediate_weights.append(weights.clone())
        V_gx, V_gy, _ = self.gradient_calc.compute_gradient(V_gray)
        U_gx, U_gy, _ = self.gradient_calc.compute_gradient(U_gray)

        for k in range(self.num_stages):
            data_fidelity_grad = (W1 * (Fuse - U_gray) + W2 * (Fuse - V_gray) - self.beta_params[k] * Fuse)
            F_gd = Fuse - self.eta[k] * data_fidelity_grad

            if self.fixed_weights:
                F_prox = self.proximal_blocks[k](F_gd, U_gx, U_gy, V_gx, V_gy, self.lambda_params[k])
            else:
                F_prox = self.proximal_blocks[k](F_gd, U_gx, U_gy, V_gx, V_gy, self.lambda_params[k],
                                                 W1_pred=W1, W2_pred=W2, U=U_gray, V=V_gray)
            Fuse = F_prox
            self.intermediate_results.append(Fuse.clone())

            if not self.fixed_weights and k < self.num_stages - 1:
                weights = self.weight_net(torch.cat([U_gray, V_gray], dim=1))
                W1, W2 = weights[:, 0:1], weights[:, 1:2]
                self.intermediate_weights.append(weights.clone())

        if self.use_color_transfer and original_V is not None and original_V.shape[1] == 3:
            final_fused = self.color_transfer(Fuse, original_V)
        else:
            final_fused = Fuse.repeat(1, 3, 1, 1) if (original_V is not None and original_V.shape[1] == 3) else Fuse

        return final_fused, self.intermediate_results, self.intermediate_weights