
import os
from get_data import get_meta_data_for_diffusion

from tabddpm_reconstruction_attack import train_diffusion_for_reconstruction, reconstruct_data_categorical


def _needs_training(artifact_dir: str, cfg: dict) -> bool:
    """Return True if the model should be trained.

    Trains when:
      - model_ckpt.pkl is missing (always necessary), OR
      - attack_params["retrain"] is explicitly True (force retrain)
    Skips when checkpoint exists and retrain is not forced.
    """
    checkpoint = os.path.join(artifact_dir, "model_ckpt.pkl")
    force = cfg.get("attack_params", {}).get("retrain", False)
    return force or not os.path.exists(checkpoint)



def partial_tabddpm_reconstruction(cfg, synth, targets, qi, hidden_features):
    artifact_dir = cfg["dataset"]["dir"] + "/partial_tabddpm_artifacts"
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features)
    reconstruction = reconstruct_data_categorical(cfg, targets, qi, hidden_features)
    return reconstruction, None, None


def repaint_reconstruction(cfg, synth, targets, qi, hidden_features):
    artifact_dir = cfg["dataset"]["dir"] + "/repaint_artifacts"
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=True)
    reconstruction = reconstruct_data_categorical(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=True)
    return reconstruction, None, None


def conditioned_repaint_reconstruction(cfg, synth, targets, qi, hidden_features):
    """Hybrid: QI-conditioned training (same as TabDDPM) + RePaint sampling.

    Shares artifact directory with partial_tabddpm_reconstruction — both use
    identical QI-conditioned training, so the checkpoint can be reused.
    If TabDDPM has already run for this sample, training is skipped automatically.
    """
    artifact_dir = cfg["dataset"]["dir"] + "/partial_tabddpm_artifacts"
    cfg["dataset"]["artifacts"] = artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    meta, domain = get_meta_data_for_diffusion(cfg)
    if _needs_training(artifact_dir, cfg):
        train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=False)
    reconstruction = reconstruct_data_categorical(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=True)
    return reconstruction, None, None

