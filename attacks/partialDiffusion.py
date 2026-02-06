# import attacks
import sys
import os
from get_data import get_meta_data_for_diffusion

on_server = len(sys.argv) > 1 and sys.argv[1] == "T"
if on_server:
    sys.path.append('/home/golobs/MIA_on_diffusion/')
    sys.path.append('/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM')
else:
    sys.path.append('/Users/stevengolob/PycharmProjects/MIA_on_diffusion/')
    sys.path.append('/Users/stevengolob/PycharmProjects/MIA_on_diffusion/midst_models/single_table_TabDDPM')
from tabddpm_reconstruction_attack import train_diffusion_for_reconstruction, reconstruct_data





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

