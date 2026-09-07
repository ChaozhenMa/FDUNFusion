# test.py
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from PIL import Image
import warnings
from tqdm import tqdm 

from config import get_model_config, get_path_config, initialize_directories, print_experiment_info
from dataset import InfraredVisibleDataset
from model import InfraredVisibleFusionDUN
from train import load_or_create_model

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def create_weight_analysis_selected(weights_list, infrared_imgs, visible_imgs, fused_images, save_dir):
    num_samples = min(len(weights_list), len(infrared_imgs), 4)
    if num_samples == 0: return

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)

    for i in range(num_samples):
        U_np, V_np = infrared_imgs[i], visible_imgs[i]
        W1, W2 = weights_list[i][0, 0].cpu().numpy(), weights_list[i][0, 1].cpu().numpy()

        axes[i, 0].imshow(np.clip(U_np, 0, 1))
        axes[i, 0].set_title(f'Infrared {i + 1}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(np.clip(V_np, 0, 1))
        axes[i, 1].set_title(f'Visible {i + 1}')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(W1)
        axes[i, 2].set_title(f'W1 {i + 1}')
        axes[i, 2].axis('off')
        axes[i, 3].imshow(W2)
        axes[i, 3].set_title(f'W2 {i + 1}')
        axes[i, 3].axis('off')
        axes[i, 4].imshow(fused_images[i])
        axes[i, 4].set_title(f'Fused {i + 1}')
        axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_analysis_selected.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_grid(infrared_imgs, visible_imgs, fused_imgs, save_dir, max_display_size=512):
    num_imgs = len(infrared_imgs)
    if num_imgs == 0: return

    sample_height, sample_width = infrared_imgs[0].shape[:2]
    scale = min(max_display_size / sample_height,
                max_display_size / sample_width) if sample_height > max_display_size or sample_width > max_display_size else 1
    new_height, new_width = int(sample_height * scale), int(sample_width * scale)

    rows, cols = min(4, (num_imgs + 2) // 3), min(3, num_imgs)
    fig, axes = plt.subplots(rows, cols * 3, figsize=(6 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif rows > 1 and cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_imgs:
                ax_ir = axes[i, j * 3] if rows > 1 else axes[j * 3]
                ax_vis = axes[i, j * 3 + 1] if rows > 1 else axes[j * 3 + 1]
                ax_fused = axes[i, j * 3 + 2] if rows > 1 else axes[j * 3 + 2]

                ir_img = Image.fromarray((infrared_imgs[idx] * 255).astype(np.uint8)).resize(
                    (new_width, new_height)) if scale != 1 else infrared_imgs[idx]
                vis_img = Image.fromarray((visible_imgs[idx] * 255).astype(np.uint8)).resize(
                    (new_width, new_height)) if scale != 1 else visible_imgs[idx]
                fused_img = Image.fromarray((fused_imgs[idx] * 255).astype(np.uint8)).resize(
                    (new_width, new_height)) if scale != 1 else fused_imgs[idx]

                ax_ir.imshow(np.array(ir_img))
                ax_ir.set_title(f'Infrared {idx + 1}')
                ax_ir.axis('off')
                ax_vis.imshow(np.array(vis_img))
                ax_vis.set_title(f'Visible {idx + 1}')
                ax_vis.axis('off')
                ax_fused.imshow(np.array(fused_img))
                ax_fused.set_title(f'Fused {idx + 1}')
                ax_fused.axis('off')
            else:
                for k in range(3): (axes[i, j * 3 + k] if rows > 1 else axes[j * 3 + k]).axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_weight_images(weights, save_dir, batch_idx):
    W1, W2 = weights[0].cpu().numpy(), weights[1].cpu().numpy()
    plt.imsave(os.path.join(save_dir, 'weights', f'W1_{batch_idx + 1:02d}.png'), W1, cmap='viridis')
    plt.imsave(os.path.join(save_dir, 'weights', f'W2_{batch_idx + 1:02d}.png'), W2, cmap='viridis')


def visualize_results(model, test_loader, save_dir, save_all=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    os.makedirs(os.path.join(save_dir, 'individual'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'weights'), exist_ok=True)

    all_fused_images, all_infrared_images, all_visible_images, all_weights = [], [], [], []
    total_processed = 0

    print("\n开始执行图像融合与可视化...")
    with torch.no_grad():
        # ✅ 新增：在这里套上 tqdm 进度条
        for batch_idx, (U_list, V_list) in enumerate(tqdm(test_loader, desc="🚀 图像融合进度", unit="batch")):
            for i in range(len(U_list)):
                U, V = U_list[i].unsqueeze(0).to(device), V_list[i].unsqueeze(0).to(device)
                fused, _, weights_list = model(U, V)
                weights = weights_list[0]
                all_weights.append(weights.cpu())

                U_np = np.clip(U[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
                V_np = np.clip(V[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
                fused_np = np.clip(fused[0].cpu().permute(1, 2, 0).numpy(), 0, 1)

                Image.fromarray((fused_np * 255).astype(np.uint8)).save(
                    os.path.join(save_dir, 'individual', f'fused_{total_processed + 1}.png'))
                save_weight_images(weights[0], save_dir, total_processed)

                all_fused_images.append(fused_np)
                all_infrared_images.append(U_np)
                all_visible_images.append(V_np)
                total_processed += 1

            if not save_all and total_processed >= 44: break

    if all_fused_images:
        create_comparison_grid(all_infrared_images[:12], all_visible_images[:12], all_fused_images[:12], save_dir)
        create_weight_analysis_selected(all_weights[:4], all_infrared_images[:4], all_visible_images[:4],
                                        all_fused_images[:12], save_dir)
    print(f"\n✅ 成功评估并保存 {total_processed} 张图像到 {save_dir}")


def custom_collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def main():
    parser = argparse.ArgumentParser(description='FDUNFusion 测试推理脚本')
    parser.add_argument('--order', type=float, default=0.1, help='指定测试模型的 order 参数 (默认: 0.1)')
    parser.add_argument('--lambda_init', type=float, default=0.9, help='指定测试模型的 lambda_init 参数 (默认: 0.9)')
    parser.add_argument('--beta_init', type=float, default=0.3, help='指定测试模型的 beta_init 参数 (默认: 0.3)')
    parser.add_argument('--ir_test_dir', type=str, default=None, help='红外测试集路径 (覆盖 config)')
    parser.add_argument('--vis_test_dir', type=str, default=None, help='可见光测试集路径 (覆盖 config)')
    args = parser.parse_args()

    exp_overrides = {
        'order': args.order,
        'lambda_init': args.lambda_init,
        'beta_init': args.beta_init,
    }

    path_overrides = exp_overrides.copy()
    if args.ir_test_dir: path_overrides['ir_test_dir'] = args.ir_test_dir
    if args.vis_test_dir: path_overrides['vis_test_dir'] = args.vis_test_dir

    torch.manual_seed(42)
    np.random.seed(42)

    initialize_directories(**path_overrides)
    print("\n[测试模式] ")
    print_experiment_info(**path_overrides)

    path_config = get_path_config(**path_overrides)
    model_config = get_model_config(**exp_overrides)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = InfraredVisibleDataset(ir_dir=path_config['ir_test_dir'], vis_dir=path_config['vis_test_dir'],
                                          transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    # ==================== 智能路径检索逻辑 ====================
    final_model_dir = "model"
    model_base_name = path_config['model_name'].replace('.pth', '')
    model_path = None

    if os.path.exists(final_model_dir):
        existing_final_models = [f for f in os.listdir(final_model_dir) if
                                 f.startswith(model_base_name) and f.endswith('.pth')]
        if existing_final_models:
            existing_final_models.sort()
            model_path = os.path.join(final_model_dir, existing_final_models[-1])
            print(f"🎯 优先检测到 'model' 文件夹中的最优模型: {model_path}")

    if model_path is None:
        model_path = os.path.join(path_config['checkpoint_dir'], path_config['model_name'])
        print(f"🔍 未在 'model' 文件夹中发现权重，将尝试加载常规检查点路径: {model_path}")
    # ==========================================================

    model, start_epoch, _, _ = load_or_create_model(model_path, model_class=InfraredVisibleFusionDUN,
                                                    resume_training=True, **model_config)

    if start_epoch == 0:
        print("⚠️ 警告：未在任何指定路径找到已训练的模型权重文件，当前模型将以初始未训练状态执行测试！")

    visualize_results(model, test_loader, save_dir=path_config['result_dir'], save_all=True)


if __name__ == "__main__":
    main()
