# NLP Final Project: Using GANs for Human Choice Prediction in Language-based Persuasion Games

### Gur Keinan 213635899 and Idan Pipano 213495260

## Project Description

In this project, we aim to predict human choices in language-based persuasion games.
We will use a GAN-based model to generate human-like interactions and decision making in persuasion games.
Later on we will use the generated data to train a model that predicts human choices in persuasion games, 
along with real human data and the simulated data used in the paper "Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation" by Eilam Shapira, Reut Apel, Moshe Tennenholtz and Roi Reichart.

## Files Description

We added the following files to the project:

1. `Creating_interactions_data_for_GAN_training.py` - This file is used to create the data for the GAN training. It processes the data in the file `data/games_clean_X.csv` and `EFs_by_GPT35.csv`.
2. `mega_games_list.pkl` - This file contains the data for the GAN training, so one wouldn't have to run the `Creating_interactions_data_for_GAN_training.py` file repeatedly.  
3. `train_GAN.py` - This file is used to train the GAN model. It uses the data from the `mega_games_list.pkl` file, create a dataset and dataloader for it, define the discriminator and generator architectures and train the GAN model.
4. `generator.pth`, `discriminator.pth` - These files contain the trained GAN models so one wouldn't have to run the `train_GAN.py` file repeatedly.
5. `creating_interactions_from_GAN_output.py` - This file is used to create the data for the human choice prediction model. It contains function that transform the GAN output to the same format as the data used in training. Read in our mini-paper what are the needed transformations.
6. `generate_using_GAN.py` - This file is used to generate data using the trained GAN model. It uses the trained generator model and generates data for the human choice prediction model. It defines the dataset to be used in the training.

In addition, we changed the file `environments/environment` to include a training phase that uses the GAN generated data in addition to the real human data and the simulated data used in the paper.
Finally, we changed the file `StrategyTransfer.py` and `RunningScripts/final_sweep_213495260_213635899.py` to include additional hyperparameters for the running and the sweep.

## Running the Code

First, create the conda environment and activate it:
```bash
conda env create -f requirements.yml
conda activate final_project_env
```

To fully reproduce the results without retraining the GAN models, run the following command
```bash
python StrategyTransfer.py
```

To reproduce the results with retraining the GAN models, run the following command:
```bash
python StrategyTransfer.py --train_gan yes
```
