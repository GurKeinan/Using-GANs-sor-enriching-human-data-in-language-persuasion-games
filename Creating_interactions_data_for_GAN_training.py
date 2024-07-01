import pandas as pd
import pickle

# divisions to topics based on https://www.jair.org/index.php/jair/article/view/13510/26781
POSITIVE_PART_TOPICS = ['positive_facilities', 'positive_price', 'positive_design', 'positive_location',
                        'positive_room', 'positive_staff', 'positive_food', 'positive_view', 'positive_transportation',
                        'positive_sanitary_facilities',
                        ]
# without 11 - nothing positive
POSITIVE_PART_PROPERTIES = ['is_positive_empty', 'has_positive_summary',
                            'positive_group_1', 'positive_group_2', 'positive_group_3',
                            'positive_0_99_chars', 'positive_100_199_chars', 'positive_200_inf_chars']

NEGATIVE_PART_TOPICS = ['negative_price', 'negative_staff', 'negative_sanitary_facilities', 'negative_room',
                        'negative_food', 'negative_location', 'negative_facilities', 'negative_air']

# without 29 - nothing negative
NEGATIVE_PART_PROPERTIES = ['is_negative_empty', 'has_negative_summary', 'negative_group_1', 'negative_group_2',
                            'negative_group_3', 'negative_0_99_chars', 'negative_100_199_chars',
                            'negative_200_inf_chars']

# without 37-39 - detailed review, review structured as a list, positive part shown first
OVERALL_REVIEW_PROPERTIES = ['positive_negative_ratio_0_0.7',
                             'positive_negative_ratio_0.7_4', 'positive_negative_ratio_4_inf']

ENGINEERED_REVIEW_FEATURES = (POSITIVE_PART_TOPICS + POSITIVE_PART_PROPERTIES + NEGATIVE_PART_TOPICS +
                              NEGATIVE_PART_PROPERTIES + OVERALL_REVIEW_PROPERTIES)

# Read the CSV files
with open('data/games_clean_X.csv', 'r') as f:
    games = pd.read_csv(f)

with open('data/EFs_by_GPT35.csv', 'r') as f:
    EFs = pd.read_csv(f, index_col=0)

# Extract unique user IDs
users_ids = games['user_id'].unique()

mega_games_list = []
for user_id in users_ids:
    # for each user, extract the games they played and divide them by bot strategy
    user_games = games[games['user_id'] == user_id]
    bots_played_by_user = user_games['strategy_id'].unique()
    for strategy in bots_played_by_user:
        user_rounds_with_strategy = user_games[user_games['strategy_id'] == strategy]
        strategy_rounds_info = []
        for index, round in user_rounds_with_strategy.iterrows():
            review_id = round['reviewId']

            # Check if review_id exists in EFs DataFrame
            if review_id in EFs.index:
                # Extract the engineered features for the review and give them the right names
                round_info_dict = {}
                review_efs = list(EFs.loc[review_id])
                for ef_index, ef in enumerate(review_efs):
                    round_info_dict[ENGINEERED_REVIEW_FEATURES[ef_index]] = ef
                round_info_dict['reaction_time'] = round['reaction_time']
                round_info_dict['didGo'] = round['didGo']
                round_info_dict['hotels_scores'] = round['hotelScore']

                strategy_rounds_info.append(round_info_dict)
            else:
                print(f"Review ID {review_id} not found in EFs DataFrame.")

        mega_games_list.append(strategy_rounds_info)

# Save the list of games
with open('mega_games_list.pkl', 'wb') as f:
    pickle.dump(mega_games_list, f)

print(mega_games_list[0])
