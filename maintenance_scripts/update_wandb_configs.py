# import wandb
# api = wandb.Api()
# # Replace with your entity, project, and run ID
# run = api.run("golobs-university-of-washington/tabular-reconstruction-attacks/cwhivcz7")
# run.config["attack_params"]["method_ref"] = "attention_chained_reconstruction"
# run.config["attack_method"] = "Attention"
# run.tags = ["NIST", "development", "chained"]
# run.update()


import wandb

api = wandb.Api()

entity = "golobs-university-of-washington"
project = "tabular-reconstruction-attacks"

# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")

for run in runs:
    print(run)
    attack_method = run.config.get("attack_method")

    if attack_method in ["TabDDPM+Repaint", "TabDDPM+RePaint"]:
        print(f"Updating run {run.id} ({attack_method} → RePaint)")

        run.config["attack_method"] = "RePaint"
        run.update()

print("Done.")
