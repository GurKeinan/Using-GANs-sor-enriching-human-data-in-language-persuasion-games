from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

from tqdm import tqdm

GENERATOR_PATH = 'generator.pth'
DISCRIMINATOR_PATH = 'discriminator.pth'

# set device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# divisions to topics based on https://www.jair.org/index.php/jair/article/view/13510/26781

POSITIVE_BINARY_FEATURES = ['postive_facilities', 'positive_price', 'positive_design', 'positive_location',
                            'positive_room', 'positive_staff', 'positive_food', 'positive_view',
                            'positive_transportation', 'positive_sanitary_facilities', 'is_positive_empty',
                            'has_positive_summary']
POSITIVE_GROUP_FEATURES = ['positive_group_1', 'positive_group_2', 'positive_group_3']
POSITIVE_LENGTH_FEATURES = ['positive_0_99_chars', 'positive_100_199_chars', 'positive_200_inf_chars']

NEGATIVE_BINARY_FEATURES = ['negative_price', 'negative_staff', 'negative_sanitary_facilities', 'negative_room',
                            'negative_food', 'negative_location', 'negative_facilities', 'negative_air',
                            'is_negative_empty', 'has_negative_summary']

NEGATIVE_GROUP_FEATURES = ['negative_group_1', 'negative_group_2', 'negative_group_3']

NEGATIVE_LENGTH_FEATURES = ['negative_0_99_chars', 'negative_100_199_chars', 'negative_200_inf_chars']

OVERALL_RATIO_FEATURES = ['positive_negative_ratio_0_0.7', 'positive_negative_ratio_0.7_4',
                          'positive_negative_ratio_4_inf']

REACTION_TIME_FEATURE = ['reaction_time_0_400', 'reaction_time_400_800', 'reaction_time_800_1200',
                         'reaction_time_1200_1600', 'reaction_time_1600_2500', 'reaction_time_2500_4000',
                         'reaction_time_4000_6000', 'reaction_time_6000_12000', 'reaction_time_12000_20000',
                         'reaction_time_20000_inf']

ENGINEERED_REVIEW_FEATURES = (POSITIVE_BINARY_FEATURES + POSITIVE_GROUP_FEATURES + POSITIVE_LENGTH_FEATURES +
                              NEGATIVE_BINARY_FEATURES + NEGATIVE_GROUP_FEATURES + NEGATIVE_LENGTH_FEATURES +
                              OVERALL_RATIO_FEATURES + ['didGo', 'hotels_scores'] + REACTION_TIME_FEATURE)

BINARY_FEATURES = POSITIVE_BINARY_FEATURES + NEGATIVE_BINARY_FEATURES + POSITIVE_GROUP_FEATURES + NEGATIVE_GROUP_FEATURES + [
    'didGo']
ONLY_ONE_CAN_BE_ONE_FEATURE_GROUPS = [POSITIVE_LENGTH_FEATURES, NEGATIVE_LENGTH_FEATURES, OVERALL_RATIO_FEATURES,
                                      REACTION_TIME_FEATURE]


def list_of_dicts_to_tensor(dict_list, padding_length=0):
    tensors_list = []
    # pad the tensor with zeros
    for _ in range(padding_length - len(dict_list)):
        tensors_list += [torch.zeros(49)]
    for d in dict_list:
        reaction_time = d.pop('reaction_time')
        d['reaction_time_0_400'] = 1 if reaction_time <= 400 else 0
        d['reaction_time_400_800'] = 1 if 400 < reaction_time <= 800 else 0
        d['reaction_time_800_1200'] = 1 if 800 < reaction_time <= 1200 else 0
        d['reaction_time_1200_1600'] = 1 if 1200 < reaction_time <= 1600 else 0
        d['reaction_time_1600_2500'] = 1 if 1600 < reaction_time <= 2500 else 0
        d['reaction_time_2500_4000'] = 1 if 2500 < reaction_time <= 4000 else 0
        d['reaction_time_4000_6000'] = 1 if 4000 < reaction_time <= 6000 else 0
        d['reaction_time_6000_12000'] = 1 if 6000 < reaction_time <= 12000 else 0
        d['reaction_time_12000_20000'] = 1 if 12000 < reaction_time <= 20000 else 0
        d['reaction_time_20000_inf'] = 1 if 20000 < reaction_time else 0
        tensors_list += [torch.tensor([int(v) if isinstance(v, bool) else v for v in d.values()])]
    return torch.stack(tensors_list)


class ReviewsDataset(Dataset):
    def __init__(self, mega_games_list: List[List[Dict[str, Any]]]):
        # TODO: check how many records remain
        self.complete_games_data = mega_games_list
        self.data = []
        for game in self.complete_games_data:
            number_of_mini_games_to_extract = len(game) // 10  # we might want to change this to 1
            for mini_game_index in range(number_of_mini_games_to_extract):
                rounds_of_minigame = game[mini_game_index * 10: (mini_game_index + 1) * 10]
                self.data.append(rounds_of_minigame)

        a = 2

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        dict_item: dict = self.data[idx]
        return list_of_dicts_to_tensor(dict_item, 0)


import torch
import torch.nn.functional as F


def gumbel_sigmoid(logits, tau=.1):
    """
    Sample from the Gumbel-Sigmoid distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        tau: non-negative scalar temperature
    Returns:
        [batch_size, n_class] sample from the Gumbel-Sigmoid distribution.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    y = torch.sigmoid(y / tau)

    return y


def gumbel_softmax(logits, tau=.1, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        tau: non-negative scalar temperature
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    y = F.softmax(y / tau, dim=-1)

    if hard:
        # Straight-through trick
        y_hard = torch.argmax(y, dim=-1)
        y_hard = F.one_hot(y_hard, num_classes=logits.size(-1)).float()
        y = (y_hard - y).detach() + y

    return y


class LSTMDiscriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        out, _ = self.lstm(x)  # h_0 and c_0 default to zero if not provided
        out = out[-1, :, :]
        out = self.fc(out).flatten()  # now out is of dimension B
        out = self.sigmoid(out)
        return out


class StupidDiscriminator(torch.nn.Module):
    def __init__(self):
        super(StupidDiscriminator, self).__init__()
        self.fc = torch.nn.Linear(49, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x[:, -1, :].double()).flatten()  # TODO: maybe .flatten()
        out = self.sigmoid(out)
        return out


class LSTMGenerator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        # self.positive_binary_features_fc_list = [torch.nn.Linear(hidden_size, 1)
        #                                          for _ in POSITIVE_BINARY_FEATURES]
        self.positive_binary_features_fc = torch.nn.Linear(hidden_size, len(POSITIVE_BINARY_FEATURES))
        # self.negative_binary_features_fc_list = [torch.nn.Linear(hidden_size, 1)
        #                                          for _ in NEGATIVE_BINARY_FEATURES]
        self.negative_binary_features_fc = torch.nn.Linear(hidden_size, len(NEGATIVE_BINARY_FEATURES))
        # self.positive_group_fcs = [torch.nn.Linear(hidden_size, 1)
        #                            for _ in POSITIVE_GROUP_FEATURES]
        # self.negative_group_fcs = [torch.nn.Linear(hidden_size, 1)
        #                               for _ in NEGATIVE_GROUP_FEATURES]
        self.positive_group_fc = torch.nn.Linear(hidden_size, len(POSITIVE_GROUP_FEATURES))
        self.negative_group_fc = torch.nn.Linear(hidden_size, len(NEGATIVE_GROUP_FEATURES))
        self.positive_length_fc = torch.nn.Linear(hidden_size, len(POSITIVE_LENGTH_FEATURES))
        self.negative_length_fc = torch.nn.Linear(hidden_size, len(NEGATIVE_LENGTH_FEATURES))
        self.overall_ratio_fc = torch.nn.Linear(hidden_size, len(OVERALL_RATIO_FEATURES))
        self.didGo_fc = torch.nn.Linear(hidden_size, 1)
        self.hotelScore_fc = torch.nn.Linear(hidden_size, 1)  # the name is incosistent with all the rest of the project
        self.reaction_time_fc = torch.nn.Linear(hidden_size, len(REACTION_TIME_FEATURE))
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, initial_input, n):
        generated_reviews = []
        current_batch_size = initial_input.shape[0]
        # input_seq = initial_input.unsqueeze(0).double().to(device)  # Ensure batch dimension is 1
        input_seq = initial_input
        h0 = torch.zeros(self.num_layers, self.hidden_size).double().to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).double().to(device)

        for _ in range(n):
            out, (h0, c0) = self.lstm(input_seq, (h0, c0))

            # positive_binary_features = torch.tensor([gumbel_sigmoid(fc(out)) for fc in
            #                                          self.positive_binary_features_fc_list])
            positive_binary_features = gumbel_sigmoid(self.positive_binary_features_fc(out))
            # negative_binary_features = torch.tensor([gumbel_sigmoid(fc(out)) for fc in
            #                                          self.negative_binary_features_fc_list])
            negative_binary_features = gumbel_sigmoid(self.negative_binary_features_fc(out))
            # positive_group_list = torch.tensor([gumbel_sigmoid(fc(out)) for fc in self.positive_group_fcs])
            # negative_group_list = torch.tensor([gumbel_sigmoid(fc(out)) for fc in self.negative_group_fcs])
            positive_group_list = gumbel_softmax(self.positive_group_fc(out))
            negative_group_list = gumbel_softmax(self.negative_group_fc(out))

            positive_length = gumbel_softmax(self.positive_length_fc(out))
            negative_length = gumbel_softmax(self.negative_length_fc(out))
            overall_ratio = gumbel_softmax(self.overall_ratio_fc(out))
            didGo = gumbel_sigmoid(self.didGo_fc(out))
            hotels_scores = self.hotelScore_fc(out)
            reaction_time = gumbel_softmax(self.reaction_time_fc(out))

            # Store generated features - should be of size (batch_size, sum of all features)
            generated_review = torch.cat((positive_binary_features,
                                          positive_group_list,
                                          positive_length,
                                          negative_binary_features,
                                          negative_group_list,
                                          negative_length,
                                          overall_ratio,
                                          didGo,
                                          hotels_scores,
                                          reaction_time), dim=-1)

            generated_reviews.append(generated_review)

            # Prepare input for the next step
            # input_seq = out

        # return a tensor of generated reviews - should be of size (n, len(POSITIVE_BINARY_FEATURES) +
        # (NEGATIVE_BINARY_FEATURES) + len(POSITIVE_GROUP_FEATURES) + len(NEGATIVE_GROUP_FEATURES) +
        # len(POSITIVE_LENGTH_FEATURES) + len(NEGATIVE_LENGTH_FEATURES) + len(OVERALL_RATIO_FEATURES) +
        # len(REACTION_TIME_FEATURE) + 2)

        list_of_reviews_tensors = [torch.tensor(review) for review in generated_reviews]
        return torch.stack(list_of_reviews_tensors).view(current_batch_size, 10, -1)

    def generate_list_of_dicts(self, initial_input, n):
        generated_reviews = self.forward(initial_input, n)
        return self.generate_list_of_dicts_from_generated_reviews(generated_reviews)

    def generate_list_of_dicts_from_generated_reviews(self, generated_reviews):
        list_of_dicts = []
        for review in generated_reviews:
            dict_review = {}
            # for i, feature in enumerate(ENGINEERED_REVIEW_FEATURES):
            #     dict_review[feature] = review[0][i].item()
            for i, feature in enumerate(ENGINEERED_REVIEW_FEATURES):
                if feature in BINARY_FEATURES:
                    dict_review[feature] = round(review[i].item())
                else:
                    dict_review[feature] = review[i].item()
            # for each group of features such that only one of the features in the group can be 1, find the index of the
            # feature that should be 1 (the feature with the highest value) and set it to 1 and the rest to 0
            for group_of_features in ONLY_ONE_CAN_BE_ONE_FEATURE_GROUPS:

                max_feature = max(group_of_features, key=dict_review.get)
                for feature in group_of_features:
                    dict_review[feature] = 1 if feature == max_feature else 0

            # sample a value for dict_review['reaction_time']
            reaction_time_true_feature = max(REACTION_TIME_FEATURE, key=dict_review.get)
            min_reaction_time, max_reaction_time = reaction_time_true_feature.split('_')[-2:]
            if max_reaction_time == 'inf':
                dict_review['reaction_time'] = torch.distributions.exponential.Exponential(1).sample().item() + int(
                    min_reaction_time)
            else:
                dict_review['reaction_time'] = torch.distributions.uniform.Uniform(int(min_reaction_time),
                                                                                   int(max_reaction_time)).sample().item()

            list_of_dicts.append(dict_review)
        return list_of_dicts


def train_GAN(generator_repetitions):
    with open('mega_games_list.pkl', 'rb') as f:
        mega_games_list = pickle.load(f)

    reviews_dataset = ReviewsDataset(mega_games_list)
    batch_size = 32
    reviews_loader = DataLoader(reviews_dataset, batch_size=batch_size, shuffle=True)
    n = 10  # number of reviews generated

    # Initialize the models
    input_size = 1
    hidden_size = 64
    num_layers = 1

    discriminator = StupidDiscriminator().to(device)
    generator = LSTMGenerator(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)

    # Initialize the optimizers
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.03)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Training loop
    num_epochs = 1

    for epoch in tqdm(range(num_epochs)):
        real_acc = 0
        fake_acc = 0
        for i, data in enumerate(reviews_loader):
            real_reviews = data.float().to(device)

            # Train the discriminator on real reviews
            discriminator_optimizer.zero_grad()
            outputs = discriminator(real_reviews)
            real_loss = criterion(outputs, torch.ones(outputs.shape[0], dtype=torch.double).to(device))
            real_acc += torch.mean((outputs > 0.5).float())

            # Generate fake reviews
            initial_input = torch.rand(batch_size, input_size).to(device)  # match the batch size
            generated_reviews = generator(initial_input, n)

            # Flatten the generated reviews to match the discriminator input shape
            generated_reviews = generated_reviews.view(batch_size, 10, -1)

            outputs = discriminator(generated_reviews)
            fake_loss = criterion(outputs, torch.zeros(outputs.shape[0], dtype=torch.double).to(device))
            fake_acc += torch.mean((outputs < 0.5).float())

            # Update discriminator weights
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train the generator
            for _ in range(generator_repetitions):
                # Generate fake reviews
                initial_input = torch.rand(batch_size, input_size).to(device)  # match the batch size
                generated_reviews = generator(initial_input, n)

                # Flatten the generated reviews to match the discriminator input shape
                generated_reviews = generated_reviews.view(batch_size, 10, -1)

                generator_optimizer.zero_grad()
                outputs = discriminator(generated_reviews)
                generator_loss = criterion(outputs, torch.ones(outputs.shape[0], dtype=torch.double).to(device))

                # Update generator weights
                generator_loss.backward()
                generator_optimizer.step()

            if (i + 1) % (print_every := 5) == 0:
                # print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(reviews_loader)}], "
                #       f"Discriminator Loss: {discriminator_loss.item()}, Generator Loss: {generator_loss.item()}, "
                #       f"Real Accuracy: {real_acc / print_every}, Fake Accuracy: {fake_acc / print_every}")
                real_acc = 0
                fake_acc = 0

    # Save the models
    torch.save(discriminator, 'discriminator.pth')
    torch.save(generator, 'generator.pth')

    return generator, discriminator



