import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import torch.optim as optim

from config import (get_model_config, get_train_config, get_path_config, get_loss_config,
                    initialize_directories, print_experiment_info)
from dataset import InfraredVisibleDataset
from model import InfraredVisibleFusionDUN
from loss import CombinedFusionLoss


def load_or_create_model(model_path=None, model_class=InfraredVisibleFusionDUN, resume_training=True, **model_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_model_config(**model_kwargs)
    model = model_class(**config).to(device)
    start_epoch, train_losses, val_losses = 0, [], []

    if not resume_training:
        print("🆕 Retraining mode: Creating new model")
        return model, start_epoch, train_losses, val_losses

    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                print(f"✅ Successfully loaded checkpoint: {model_path} (Epoch {start_epoch})")
            else:
                model.load_state_dict(
                    checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model_state_dict', checkpoint))
                start_epoch = 1
                print(f"✅ Successfully loaded model state dict: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}. Creating new model")
    else:
        print(f"🆕 Creating new model for training (not found {model_path})")

    return model, start_epoch, train_losses, val_losses


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, filepath):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(checkpoint, filepath)


def auto_train_fusion_network(model, train_loader, val_loader, **kwargs):
    # Pass through kwargs to ensure command line parameters override all sub-configurations
    train_config = get_train_config(**kwargs)
    loss_config = get_loss_config(**kwargs)
    path_config = get_path_config(**kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_config = {
        'num_stages': model.num_stages,
        'lambda_init': model.lambda_params[0].item() if hasattr(model, 'lambda_params') else 0.9,
        'beta_init': model.beta_params[0].item() if hasattr(model, 'beta_params') else 0.7,
        'fixed_weights': model.fixed_weights,
        'use_fractional': model.use_fractional,
        'order': model.gradient_calc.order if hasattr(model, 'gradient_calc') and hasattr(model.gradient_calc,
                                                                                          'order') else 0.1,
        'color_transfer': model.use_color_transfer if hasattr(model, 'use_color_transfer') else True
    }

    checkpoint_path = os.path.join(path_config['checkpoint_dir'], path_config['model_name'])

    if train_config['resume_training']:
        model, start_epoch, train_losses, val_losses = load_or_create_model(
            checkpoint_path, model_class=type(model), resume_training=True, **model_config
        )
    else:
        start_epoch, train_losses, val_losses = 0, [], []
        model = model.to(device)

    num_epochs = train_config['num_epochs']
    if num_epochs - start_epoch <= 0:
        print(f"✅ Model has completed {start_epoch} epochs of training, no need to continue training")
        return model, train_losses, val_losses

    current_order = model.gradient_calc.order if hasattr(model, 'gradient_calc') and hasattr(model.gradient_calc,
                                                                                             'order') else 0.1
    criterion = CombinedFusionLoss(use_fractional=model_config['use_fractional'], frac_order=current_order,
                                   **loss_config)
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    if train_config['resume_training'] and start_epoch > 0 and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            pass

    patience = train_config.get('early_stopping_patience', 25)
    best_val_loss = float('inf')
    patience_counter = 0
    min_delta = 1e-5

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        total_samples = 0
        total_batches = len(train_loader)

        for batch_idx, (U, V) in enumerate(train_loader):
            batch_samples = U.size(0)
            total_samples += batch_samples
            U, V = U.to(device), V.to(device)

            optimizer.zero_grad()
            fused, _, weights_list = model(U, V)
            weights = weights_list[-1]

            total_loss, _, _, _, _ = criterion(fused, U, V, weights, current_epoch=epoch)
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * batch_samples

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch + 1:02d} | Batch: {batch_idx:04d}/{total_batches:04d} | '
                      f'Batch Avg Loss: {total_loss.item():.6f} | Samples in batch: {batch_samples}')

        avg_train_loss = train_loss / total_samples if total_samples > 0 else 0
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for U_list, V_list in val_loader:
                for i in range(len(U_list)):
                    U, V = U_list[i].unsqueeze(0).to(device), V_list[i].unsqueeze(0).to(device)
                    fused, _, weights_list = model(U, V)
                    total_loss, _, _, _, _ = criterion(fused, U, V, weights_list[0])
                    val_loss += total_loss.item()
                    val_samples += 1

        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        val_losses.append(avg_val_loss)
        scheduler.step()

        print(
            f'Epoch {epoch + 1:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, checkpoint_path)
            print(f"🌟 Validation Loss reached a new low ({best_val_loss:.6f}), updated best model!")
        else:
            patience_counter += 1
            print(f"⚠️ Validation Loss did not improve significantly (Patience: {patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered! No significant decrease in Validation Loss for {patience} consecutive Epochs.")
            print(f"Best validation Loss stayed at: {best_val_loss:.6f}")
            break

        elif (epoch + 1) % train_config['save_interval'] == 0:
            backup_path = checkpoint_path.replace('.pth', f'_epoch{epoch + 1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, backup_path)

    return model, train_losses, val_losses


def custom_collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def main():
    default_model_config = get_model_config()
    default_train_config = get_train_config()
    parser = argparse.ArgumentParser(description='FDUNFusion Training Script')
    parser.add_argument('--order', type=float, default=default_model_config['order'], help='Fractional order (default: config.py)')
    parser.add_argument('--lambda_init', type=float, default=default_model_config['lambda_init'], help='Lambda initialization (default: config.py)')
    parser.add_argument('--beta_init', type=float, default=default_model_config['beta_init'], help='Beta initialization (default: config.py)')
    parser.add_argument('--batch_size', type=int, default=default_train_config['batch_size'], help='Training batch size (default: config.py)')
    parser.add_argument('--force-retrain', action='store_true', help='Ignore existing final model, force retraining')
    args = parser.parse_args()

    exp_overrides = {
        'order': args.order,
        'lambda_init': args.lambda_init,
        'beta_init': args.beta_init,
        'batch_size': args.batch_size
    }

    torch.manual_seed(42)

    initialize_directories(**exp_overrides)
    path_config = get_path_config(**exp_overrides)

    # ==================== Safe Lock Logic ====================
    # Assume your final model is stored in the 'model' folder under the root directory
    final_model_dir = "model"
    os.makedirs(final_model_dir, exist_ok=True)
    # Use endswith for fuzzy matching, or directly compose the expected filename
    # Since your best saved model name includes epoch90, we can traverse and check
    model_base_name = path_config['model_name'].replace('.pth', '')

    existing_final_models = [f for f in os.listdir(final_model_dir) if
                             f.startswith(model_base_name) and f.endswith('.pth')]

    if existing_final_models and not args.force_retrain:
        print("\n" + "=" * 60)
        print("🛑 Training Interrupted: Detected existing optimal model for this configuration in the 'model' folder!")
        for f in existing_final_models:
            print(f"   - {os.path.join(final_model_dir, f)}")
        print("\n💡 You can directly run test.py for testing and evaluation.")
        print("⚠️ If you confirm to overwrite or retrain the model, add the --force-retrain parameter, e.g.:")
        print("   python train.py --force-retrain")
        print("=" * 60 + "\n")
        return  # Directly exit the program, do not start training
    # ====================================================

    print_experiment_info(**exp_overrides)

    model_config = get_model_config(**exp_overrides)
    train_config = get_train_config(**exp_overrides)

    train_transform = transforms.Compose([transforms.RandomCrop(128), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = InfraredVisibleDataset(ir_dir=path_config['ir_train_dir'], vis_dir=path_config['vis_train_dir'],
                                           transform=train_transform)
    val_dataset = InfraredVisibleDataset(ir_dir=path_config['ir_val_dir'], vis_dir=path_config['vis_val_dir'],
                                         transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=0,
                            collate_fn=custom_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfraredVisibleFusionDUN(**model_config).to(device)

    trained_model, train_losses, val_losses = auto_train_fusion_network(model, train_loader, val_loader,
                                                                        **exp_overrides)

    if train_losses and val_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(path_config['result_dir'], 'training_loss.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
