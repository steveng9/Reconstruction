import wandb
api = wandb.Api()
# Replace with your entity, project, and run ID
run = api.run("golobs-university-of-washington/tabular-reconstruction-attacks/cwhivcz7")
run.config["attack_params"]["method_ref"] = "attention_chained_reconstruction"
run.config["attack_method"] = "Attention"
run.tags = ["NIST", "development", "chained"]
run.update()