
import os
from get_data import get_meta_data_for_diffusion

from tabddpm_reconstruction_attack import train_diffusion_for_reconstruction, reconstruct_data_categorical





def partial_tabddpm_reconstruction(cfg, synth, targets, qi, hidden_features):
    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/partial_tabddpm_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if cfg["attack_params"]["retrain"]:
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features)
    reconstruction = reconstruct_data(cfg, targets, qi, hidden_features)
    return reconstruction, None, None



def repaint_reconstruction(cfg, synth, targets, qi, hidden_features):
    cfg["dataset"]["artifacts"] = cfg["dataset"]["dir"] + "/repaint_artifacts"
    os.makedirs(cfg["dataset"]["artifacts"], exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if cfg["attack_params"]["retrain"]:
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=True)
    reconstruction = reconstruct_data(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=True)
    return reconstruction, None, None

