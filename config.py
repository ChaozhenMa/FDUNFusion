# config.py
"""
Infrared and Visible Image Fusion Network Configuration File
Centralized management of all model, training and path parameters
"""
import os

# ==================== Global Version Control ====================
GLOBAL_VERSION = "7"
GLOBAL_BASE_NAME = f"result{GLOBAL_VERSION}"

# ==================== Default Experiment Parameters ====================
# These are the default parameters when directly running train.py or test.py
DEFAULT_EXP_PARAMS = {
    'order': 0.1,  # Fractional order (0.1 corresponds to low-frequency information extraction)
    'lambda_init': 0.9,  # Physical prior weight initialization parameter lambda
    'beta_init': 0.3  # Physical prior weight initialization parameter beta
}

# ==================== Basic Configuration ====================
BASE_MODEL_CONFIG = {
    'num_stages': 10,
    'color_transfer': True,
    'fixed_weights': False,
    'use_fractional': True,
}

BASE_TRAIN_CONFIG = {
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'batch_size': 8,
    'save_interval': 10,
    'resume_training': True,
    'early_stopping_patience': 25
}

BASE_LOSS_CONFIG = {
    'alpha': 1.0,
    'weight_alpha': 1,
    'save_epochs': [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
}

BASE_PATH_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    # train
    'ir_train_dir': "./data/MSRS_Split/train/ir",
    'vis_train_dir': "./data/MSRS_Split/train/vi",
    # val
    'ir_val_dir': "./data/MSRS_Split/val/ir",
    'vis_val_dir': "./data/MSRS_Split/val/vi",
    # test
    'ir_test_dir': "./data/MSRS_Split/test/ir",
    'vis_test_dir': "./data/MSRS_Split/test/vi",
    # When testing other datasets, uncomment here or specify via command line
    # 'ir_test_dir': "E:/dataset/M3FD/ir",
    # 'vis_test_dir': "E:/dataset/M3FD/vi",
    # 'ir_test_dir': "E:/dataset/TNO/TNO44-ir",
    # 'vis_test_dir': "E:/dataset/TNO/TNO44-vi",
}


def get_experiment_params(**overrides):
    params = DEFAULT_EXP_PARAMS.copy()
    params.update(overrides)
    exp_name = f"order_{params['order']:.1f}_lambda_{params['lambda_init']:.1f}_beta_{params['beta_init']:.1f}"

    return {
        'order': params['order'],
        'lambda_init': params['lambda_init'],
        'beta_init': params['beta_init'],
        'exp_name': exp_name,
        'base_name': GLOBAL_BASE_NAME
    }


def get_model_config(**overrides):
    exp_params = get_experiment_params(**overrides)
    config = BASE_MODEL_CONFIG.copy()
    config.update({
        'order': exp_params['order'],
        'lambda_init': exp_params['lambda_init'],
        'beta_init': exp_params['beta_init']
    })
    config.update(overrides)
    return config


def get_path_config(**overrides):
    exp_params = get_experiment_params(**overrides)
    base = exp_params['base_name']
    order = exp_params['order']
    lam = exp_params['lambda_init']
    bet = exp_params['beta_init']

    config = BASE_PATH_CONFIG.copy()
    config.update({
        'result_dir': f"results{GLOBAL_VERSION}/{base}_frac{order:.1f}_lam{lam:.1f}_bet{bet:.1f}",
        'model_name': f"fusion_model{GLOBAL_VERSION}_{base}_frac{order:.1f}_lam{lam:.1f}_bet{bet:.1f}.pth",
        'debug_weights': f"results{GLOBAL_VERSION}/debug_weights{GLOBAL_VERSION}/weights_{base}_frac{order:.1f}_lam{lam:.1f}_bet{bet:.1f}"
    })
    config.update(overrides)
    return config


def get_train_config(**overrides):
    config = BASE_TRAIN_CONFIG.copy()
    config.update(overrides)
    return config


def get_loss_config(**overrides):
    exp_params = get_experiment_params(**overrides)
    config = BASE_LOSS_CONFIG.copy()
    config.update({
        'gamma': exp_params['lambda_init'],
        'mu': exp_params['beta_init']
    })
    config.update(overrides)
    return config


def initialize_directories(**overrides):
    path_config = get_path_config(**overrides)
    os.makedirs(path_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(path_config['result_dir'], exist_ok=True)
    os.makedirs(path_config['debug_weights'], exist_ok=True)


def print_experiment_info(**overrides):
    exp_params = get_experiment_params(**overrides)
    path_config = get_path_config(**overrides)
    train_config = get_train_config(**overrides)

    print(" Current Experiment Configuration:")
    print(f"   Experiment Name: {exp_params['exp_name']}")
    print(f"   Global Version: {GLOBAL_VERSION}")
    print(f"   Fractional Order: {exp_params['order']}")
    print(f"   Lambda/Gamma: {exp_params['lambda_init']}")
    print(f"   Beta/Mu: {exp_params['beta_init']}")
    print(f"   Result Directory: {path_config['result_dir']}")
    print(f"   Training Epochs: {train_config['num_epochs']}")