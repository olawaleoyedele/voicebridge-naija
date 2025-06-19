from TTS.utils.manage import ModelManager
from TTS.config.shared_configs import load_config
from TTS.trainer import Trainer
from TTS.utils.audio import AudioProcessor

# ----------------- CONFIGURATION -------------------
output_path = "./outputs/hausa_tts"
dataset_path = "./data/hausa"
config_path = "./config_hausa.json"

# ----------------- LOAD CONFIG ---------------------
# Load base config from pretrained multilingual model
model_name = "tts_models/multilingual/multi-dataset/"
manager = ModelManager()
path = manager.download_model(model_name)
config = load_config(path.config_path)

# Override paths
config.output_path = output_path
config.datasets[0]["path"] = dataset_path
config.datasets[0]["language"] = "hau"  # ISO 639-3 code for Hausa

# Optional: reduce training time during testing
config.train_steps = 10000
config.eval_steps = 1000
config.save_steps = 1000

# ----------------- AUDIO PROCESSOR -----------------
ap = AudioProcessor.init_from_config(config)

# ----------------- START TRAINING ------------------
trainer = Trainer(
    config, ap, None, True
)
trainer.fit()
