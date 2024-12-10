import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("/sinoax98-georgia-institute-of-technology/llmft-experiments/runs/16pkq2hw")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")