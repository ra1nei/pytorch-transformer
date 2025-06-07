from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 80,
        "d_model": 512,
        "datasource": 'harouzie/vi_en-translation',
        "lang_src": "English",
        "lang_tgt": "Vietnamese",

        "model_weights_base_path": "/kaggle/input/best/pytorch/default/1",

        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder_path = Path(config['model_weights_base_path'])
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder_path / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder_path = Path(config['model_weights_base_path'])
    model_filename_pattern = f"{config['model_basename']}*.pt" # Thêm .pt để chỉ tìm file pt
    weights_files = list(model_folder_path.glob(model_filename_pattern))
    
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

