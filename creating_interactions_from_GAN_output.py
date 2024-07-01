from copy import deepcopy
import torch

# Order of strategic features
STRATEGIC_FEATURES_ORDER = ['roundNum', 'hotels_scores', 'action_taken', 'is_hotel_good', 'reaction_time', 'last_reaction_time',
                            'user_points', 'bot_points', 'last_didGo_True', 'last_didGo_False', 'last_didWin_True', 'last_didWin_False',
                            'last_last_didGo_True', 'last_last_didGo_False', 'last_last_didWin_True', 'last_last_didWin_False',
                            'user_earned_more', 'user_not_earned_more', 'last_reaction_time_0_400', 'last_reaction_time_400_800',
                            'last_reaction_time_800_1200', 'last_reaction_time_1200_1600', 'last_reaction_time_1600_2500',
                            'last_reaction_time_2500_4000', 'last_reaction_time_4000_6000', 'last_reaction_time_6000_12000',
                            'last_reaction_time_12000_20000', 'last_reaction_time_20000_inf']

# Positive feature sets
POSITIVE_BINARY_FEATURES = ['postive_facilities', 'positive_price', 'positive_design', 'positive_location',
                            'positive_room', 'positive_staff', 'positive_food', 'positive_view',
                            'positive_transportation', 'positive_sanitary_facilities', 'is_positive_empty',
                            'has_positive_summary']
POSITIVE_GROUP_FEATURES = ['positive_group_1', 'positive_group_2', 'positive_group_3']
POSITIVE_LENGTH_FEATURES = ['positive_0_99_chars', 'positive_100_199_chars', 'positive_200_inf_chars']

# Negative feature sets
NEGATIVE_BINARY_FEATURES = ['negative_price', 'negative_staff', 'negative_sanitary_facilities', 'negative_room',
                            'negative_food', 'negative_location', 'negative_facilities', 'negative_air',
                            'is_negative_empty', 'has_negative_summary']
NEGATIVE_GROUP_FEATURES = ['negative_group_1', 'negative_group_2', 'negative_group_3']
NEGATIVE_LENGTH_FEATURES = ['negative_0_99_chars', 'negative_100_199_chars', 'negative_200_inf_chars']

# Overall ratio features
OVERALL_RATIO_FEATURES = ['positive_negative_ratio_0_0.7', 'positive_negative_ratio_0.7_4', 'positive_negative_ratio_4_inf']

# Reaction time features
REACTION_TIME_FEATURE = ['reaction_time_0_400', 'reaction_time_400_800', 'reaction_time_800_1200', 'reaction_time_1200_1600',
                         'reaction_time_1600_2500', 'reaction_time_2500_4000', 'reaction_time_4000_6000', 'reaction_time_6000_12000',
                         'reaction_time_12000_20000', 'reaction_time_20000_inf']

# Engineered review features
ENGINEERED_REVIEW_FEATURES = (POSITIVE_BINARY_FEATURES + POSITIVE_GROUP_FEATURES + POSITIVE_LENGTH_FEATURES +
                              NEGATIVE_BINARY_FEATURES + NEGATIVE_GROUP_FEATURES + NEGATIVE_LENGTH_FEATURES +
                              OVERALL_RATIO_FEATURES + ['didGo', 'hotels_scores'] + REACTION_TIME_FEATURE)

# Ordered review features
ORDERED_REVIEW_FEATURES = (POSITIVE_BINARY_FEATURES + POSITIVE_GROUP_FEATURES + POSITIVE_LENGTH_FEATURES +
                           NEGATIVE_BINARY_FEATURES + NEGATIVE_GROUP_FEATURES + NEGATIVE_LENGTH_FEATURES +
                           OVERALL_RATIO_FEATURES)

def get_interaction_from_efs_list(efs_rounds_list):
    """
    Extract interaction data from a list of engineered feature sets for each round.

    Args:
        efs_rounds_list (list of dict): List of dictionaries, each containing features for a single round.

    Returns:
        dict: Dictionary containing tensors for strategic features and a concatenated tensor for review features.
    """
    # Initialize variables for tracking points and actions
    user_points = 0
    bot_points = 0
    user_just_won = False
    bot_just_won = False
    didGo = None
    last_didGo_True = False
    last_didGo_False = False
    last_didWin_True = False
    last_didWin_False = False
    last_last_didGo_True = False
    last_last_didGo_False = False
    last_last_didWin_True = False
    last_last_didWin_False = False
    last_reaction_time_in_0_400 = False
    last_reaction_time_in_400_800 = False
    last_reaction_time_in_800_1200 = False
    last_reaction_time_in_1200_1600 = False
    last_reaction_time_in_1600_2500 = False
    last_reaction_time_in_2500_4000 = False
    last_reaction_time_in_4000_6000 = False
    last_reaction_time_in_6000_12000 = False
    last_reaction_time_in_12000_20000 = False
    last_reaction_time_in_20000_inf = False
    last_reaction_time = -1
    interactions_list = []

    for round_num, efs_dict in enumerate(efs_rounds_list):
        interaction_dict = deepcopy(efs_dict)
        interaction_dict['roundNum'] = round_num

        # Update points and win status based on the action taken and hotel score
        if efs_dict['didGo']:
            bot_just_won = True
            bot_points += 1
            if efs_dict['hotels_scores'] > 8:
                user_points += 1
                user_just_won = True
            else:
                user_just_won = False
        else:
            bot_just_won = False
            if efs_dict['hotels_scores'] < 8:
                user_points += 1
                user_just_won = True
            else:
                user_just_won = False

        # Update score features
        interaction_dict['user_points'] = user_points
        interaction_dict['bot_points'] = bot_points
        interaction_dict['user_earned_more'] = user_points > bot_points
        interaction_dict['user_not_earned_more'] = user_points < bot_points

        # Update 'didGo' features for the current and past rounds
        interaction_dict['last_didGo_True'] = last_didGo_True
        interaction_dict['last_didGo_False'] = last_didGo_False
        interaction_dict['last_didWin_True'] = last_didWin_True
        interaction_dict['last_didWin_False'] = last_didWin_False
        interaction_dict['last_last_didGo_True'] = last_last_didGo_True
        interaction_dict['last_last_didGo_False'] = last_last_didGo_False
        interaction_dict['last_last_didWin_True'] = last_last_didWin_True
        interaction_dict['last_last_didWin_False'] = last_last_didWin_False

        # Update reaction time features
        interaction_dict['last_reaction_time_0_400'] = last_reaction_time_in_0_400
        interaction_dict['last_reaction_time_400_800'] = last_reaction_time_in_400_800
        interaction_dict['last_reaction_time_800_1200'] = last_reaction_time_in_800_1200
        interaction_dict['last_reaction_time_1200_1600'] = last_reaction_time_in_1200_1600
        interaction_dict['last_reaction_time_1600_2500'] = last_reaction_time_in_1600_2500
        interaction_dict['last_reaction_time_2500_4000'] = last_reaction_time_in_2500_4000
        interaction_dict['last_reaction_time_4000_6000'] = last_reaction_time_in_4000_6000
        interaction_dict['last_reaction_time_6000_12000'] = last_reaction_time_in_6000_12000
        interaction_dict['last_reaction_time_12000_20000'] = last_reaction_time_in_12000_20000
        interaction_dict['last_reaction_time_20000_inf'] = last_reaction_time_in_20000_inf
        interaction_dict['last_reaction_time'] = last_reaction_time

        # Add other interaction features
        interaction_dict['action_taken'] = int(efs_dict['didGo'])
        interaction_dict['is_hotel_good'] = int(efs_dict['hotels_scores'] > 8)
        interaction_dict['reaction_time'] = efs_dict['reaction_time']
        interaction_dict['hotels_score'] = efs_dict['hotels_scores']

        # Update variables for the next round
        last_last_didGo_True = last_didGo_True
        last_last_didGo_False = last_didGo_False
        last_last_didWin_True = last_didWin_True
        last_last_didWin_False = last_didWin_False
        last_didGo_True = efs_dict['didGo']
        last_didGo_False = not efs_dict['didGo']
        last_didWin_True = user_just_won
        last_didWin_False = not user_just_won
        last_reaction_time_in_0_400 = efs_dict['reaction_time_0_400']
        last_reaction_time_in_400_800 = efs_dict['reaction_time_400_800']
        last_reaction_time_in_800_1200 = efs_dict['reaction_time_800_1200']
        last_reaction_time_in_1200_1600 = efs_dict['reaction_time_1200_1600']
        last_reaction_time_in_1600_2500 = efs_dict['reaction_time_1600_2500']
        last_reaction_time_in_2500_4000 = efs_dict['reaction_time_2500_4000']
        last_reaction_time_in_4000_6000 = efs_dict['reaction_time_4000_6000']
        last_reaction_time_in_6000_12000 = efs_dict['reaction_time_6000_12000']
        last_reaction_time_in_12000_20000 = efs_dict['reaction_time_12000_20000']
        last_reaction_time_in_20000_inf = efs_dict['reaction_time_20000_inf']
        last_reaction_time = efs_dict['reaction_time']

        interactions_list.append(interaction_dict)

    # Prepare ordered strategic and review features
    ordered_strategic_features_dict = {feature: [] for feature in STRATEGIC_FEATURES_ORDER}
    ordered_review_features_dict = {feature: [] for feature in ORDERED_REVIEW_FEATURES}

    for interaction_dict in interactions_list:
        for feature in STRATEGIC_FEATURES_ORDER:
            ordered_strategic_features_dict[feature].append(interaction_dict[feature])
        for feature in ORDERED_REVIEW_FEATURES:
            ordered_review_features_dict[feature].append(interaction_dict[feature])

    # Convert lists to tensors
    for feature in STRATEGIC_FEATURES_ORDER:
        ordered_strategic_features_dict[feature] = torch.tensor(ordered_strategic_features_dict[feature])

    ordered_review_features = torch.tensor([ordered_review_features_dict[feature] for feature in ORDERED_REVIEW_FEATURES])

    # Create the game dictionary with strategic features and review vector
    game_dict = deepcopy(ordered_strategic_features_dict)
    game_dict['review_vector'] = ordered_review_features

    return game_dict
