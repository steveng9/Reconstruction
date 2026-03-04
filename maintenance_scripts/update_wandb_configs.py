# import wandb
# api = wandb.Api()
# # Replace with your entity, project, and run ID
# run = api.run("golobs-university-of-washington/tabular-reconstruction-attacks/cwhivcz7")
# run.config["attack_params"]["method_ref"] = "attention_chained_reconstruction"
# run.config["attack_method"] = "Attention"
# run.tags = ["NIST", "development", "chained"]
# run.update()


import time
import wandb
from requests.exceptions import HTTPError

api = wandb.Api(timeout=60)

entity = "golobs-university-of-washington"
project = "tabular-reconstruction-attacks"

# ── Rename broken TabDDPM adult runs ─────────────────────────────────────────
# The "main attack sweep 1" runs for TabDDPM / ConditionedRePaint / TabDDPMWithMLP
# on the adult dataset used a cached model trained with different hyperparameters
# (200k epochs / 2000 timesteps) than the production defaults (100k / 1000).
# Mark them so they're not mistaken for valid production results.

OLD_GROUP   = "main attack sweep 1"
NEW_GROUP   = "main attack sweep 1 - BROKEN"
DATASET     = "adult"
BAD_ATTACKS = {"TabDDPM", "ConditionedRePaint", "TabDDPMWithMLP", "TabDDPMEnsemble"}

runs = api.runs(
    f"{entity}/{project}",
    filters={"group": OLD_GROUP},
    per_page=50,
)

n_scanned = 0
n_updated = 0
for run in runs:
    n_scanned += 1
    if n_scanned % 50 == 0:
        print(f"  ... scanned {n_scanned} runs, updated {n_updated} so far", flush=True)

    for attempt in range(5):
        try:
            matches = (run.config.get("dataset") == DATASET
                       and run.config.get("attack_method") in BAD_ATTACKS)
            break
        except HTTPError as e:
            print(f"  [retry {attempt+1}/5] {run.id}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        print(f"  [SKIP] {run.id}: failed after 5 attempts", flush=True)
        continue

    if matches:
        for attempt in range(5):
            try:
                run.group = NEW_GROUP
                run.update()
                break
            except HTTPError as e:
                print(f"  [retry {attempt+1}/5] update {run.id}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  [SKIP] {run.id}: update failed after 5 attempts", flush=True)
            continue
        n_updated += 1
        print(f"  [{n_updated}] {run.id}  {run.config.get('attack_method')}"
              f"  sdg={run.config.get('sdg_method')}"
              f"  sample={run.config.get('sample_idx')}", flush=True)

print(f"Done. Scanned {n_scanned} runs, updated {n_updated} → '{NEW_GROUP}'")
