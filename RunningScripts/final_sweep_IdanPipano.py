import wandb
import subprocess

YOUR_WANDB_USERNAME = "idanpipano"
project = "NLP2024_PROJECT_213495260_213635899"

for seed_value in range(1, 6):
    sweep_config = {
        "name": f"GAN_simulations and generate_best_out_of (seed={seed_value})",
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "AUC.test.max"
        },
        "parameters": {
            "ENV_HPT_mode": {"values": [False]},
            "architecture": {"values": ["LSTM"]},
            "seed": {"values": [seed_value]},
            "online_simulation_factor": {"values": [4]},
            "features": {"values": ["EFs"]},
            "GAN_simulations": {"values": [0.5, 1, 2]},
            "generator_repetitions": {"values": [40]},
            "generate_best_out_of": {"values": [5, 10, 20]}
        },
        "command": [
            "${ENVIRONMENT_VARIABLE}",
            "${interpreter}",
            "StrategyTransfer.py",
            "${project}",
            "${args}"
        ]
    }

    # Initialize a new sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)
    print("run this line to run your agent in a screen:")
    command_to_run = f"screen -dmS \"sweep_agent_seed{seed_value}\" wandb agent west-best-dorms/NLP2024_PROJECT_213495260_213635899/{sweep_id}"
    print(command_to_run)
    # subprocess.run(command_to_run, shell=True)
