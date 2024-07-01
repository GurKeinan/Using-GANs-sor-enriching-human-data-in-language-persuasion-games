import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
from creating_interactions_from_GAN_output import get_interaction_from_efs_list
from train_GAN import train_GAN, LSTMDiscriminator, LSTMGenerator

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GANDataset(Dataset):
    """
    Custom Dataset for generating interactions using a GAN.
    """
    def __init__(self, generator: LSTMGenerator, discriminator: LSTMDiscriminator, simulation_size: int,
                 generate_best_out_of: int, precreated_interactions_path: str = None, n: int = 10):
        """
        Initialize the GANDataset.

        Args:
            generator (LSTMGenerator): Pretrained generator model.
            discriminator (LSTMDiscriminator): Pretrained discriminator model.
            simulation_size (int): Number of interactions to simulate.
            generate_best_out_of (int): Number of interactions to generate and select the best from.
            precreated_interactions_path (str, optional): Path to pre-created interactions CSV file.
            n (int): Number of rounds in each game interaction.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.simulation_size = simulation_size
        self.generate_best_out_of = generate_best_out_of
        self.precreated_interactions_path = precreated_interactions_path
        self.n = n

        # Load precreated interactions if a path is provided
        if precreated_interactions_path:
            self.interactions = pd.read_csv(precreated_interactions_path)
            self.simulation_size = len(self.interactions)

    def __len__(self):
        """
        Return the number of interactions in the dataset.

        Returns:
            int: Number of interactions.
        """
        return self.simulation_size

    def __getitem__(self, idx):
        """
        Get an interaction by index.

        Args:
            idx (int): Index of the interaction.

        Returns:
            dict: Interaction data.
        """
        if self.precreated_interactions_path:
            return self.interactions.iloc[idx]
        else:
            game_dict = self.generate_interaction()
            game_dict['user_id'] = idx
            game_dict['game_id'] = idx
            game_dict['bot_strategy'] = torch.tensor([0] * 10)
            game_dict['is_sample'] = torch.tensor([True] * 10)
            game_dict['weight'] = torch.tensor([1] * 10)  # TODO: Verify if this is appropriate
            game_dict['action_id'] = torch.tensor([-1] * 10)  # TODO: Confirm if always -1 for simulation
            game_dict['n_rounds'] = 10
            return game_dict

    def generate_interaction(self):
        """
        Generate an interaction by sampling multiple and selecting the best one.

        Returns:
            dict: Best interaction data.
        """
        # Generate multiple interactions and select the best one based on discriminator score
        initial_input = torch.rand(self.generate_best_out_of, 1).to(device)
        generated_reviews = self.generator(initial_input, self.n)
        scores = self.discriminator(generated_reviews)
        best_idx = torch.argmax(scores).item()
        return get_interaction_from_efs_list(
            self.generator.generate_list_of_dicts_from_generated_reviews(generated_reviews[best_idx])
        )

def get_GAN_Dataset(simulation_size: int, generator_repetitions: int = 20, generate_best_out_of: int = 5, retrain: bool = False)\
        -> GANDataset:
    """
    Get the GAN Dataset, training the GAN if necessary.

    Args:
        simulation_size (int): Number of interactions to simulate.
        generator_repetitions (int): Number of times the generator is trained per discriminator update.
        generate_best_out_of (int): Number of interactions to generate and select the best from.
        retrain (bool): Whether to retrain the GAN or not.

    Returns:
        GANDataset: Initialized dataset object.
    """
    # Load pretrained models if they exist, otherwise train new models
    if os.path.exists('generator.pth') and os.path.exists('discriminator.pth') and not retrain:
        generator = torch.load('generator.pth').to(device)
        discriminator = torch.load('discriminator.pth').to(device)
        generator.eval()
        discriminator.eval()
    else:
        generator, discriminator = train_GAN(generator_repetitions)
        generator.eval()
        discriminator.eval()

    return GANDataset(generator, discriminator, simulation_size, generate_best_out_of)
