import pandas as pd
import numpy as np

# ------------------------------
# Momentum Feature Labeler
# ------------------------------

def add_momentum_features(df):
    df = df.copy()

    df['own_consecutive_makes'] = 0
    df['opponent_consecutive_misses'] = 0
    df['own_scoring_streak'] = 0
    df['opponent_scoring_streak'] = 0
    df['last_5_play_points'] = 0
    df['pace_last_5'] = 0.0
    df['fouls_drawn_last_5'] = 0

    results = []

    for game_id, game_df in df.groupby("game_id"):
        game_df = game_df.sort_values("play_id").reset_index(drop=True)

        own_streak = 0
        opponent_streak = 0
        own_makes = 0
        opponent_misses = 0

        last_5_points = []
        last_5_lengths = []
        last_5_fouls = []

        last_team = None

        for idx, row in game_df.iterrows():
            action_team = row['action_team']
            shot_outcome = row['shot_outcome']
            scoring_play = row['scoring_play']
            foul = row['foul']
            points = 0

            # Calculate points scored
            if row['home_score'] + row['away_score'] > 0:
                if idx > 0:
                    points = (row['home_score'] + row['away_score']) - (game_df.loc[idx - 1, 'home_score'] + game_df.loc[idx - 1, 'away_score'])

            # Update streaks
            if scoring_play == 1:
                if action_team == last_team:
                    own_streak += 1
                else:
                    own_streak = 1
                    opponent_streak = 0
                own_makes += 1
                opponent_misses = 0
            else:
                if action_team != last_team:
                    opponent_streak += 1
                else:
                    opponent_streak = 0
                own_makes = 0
                opponent_misses += 1

            # Update last 5 plays trackers
            last_5_points.append(points)
            last_5_lengths.append(row['play_length'])
            last_5_fouls.append(foul)

            if len(last_5_points) > 5:
                last_5_points.pop(0)
                last_5_lengths.pop(0)
                last_5_fouls.pop(0)

            game_df.loc[idx, 'own_consecutive_makes'] = own_makes
            game_df.loc[idx, 'opponent_consecutive_misses'] = opponent_misses
            game_df.loc[idx, 'own_scoring_streak'] = own_streak
            game_df.loc[idx, 'opponent_scoring_streak'] = opponent_streak
            game_df.loc[idx, 'last_5_play_points'] = sum(last_5_points)
            game_df.loc[idx, 'pace_last_5'] = np.mean(last_5_lengths) if last_5_lengths else 0
            game_df.loc[idx, 'fouls_drawn_last_5'] = sum(last_5_fouls)

            last_team = action_team

        results.append(game_df)

    return pd.concat(results).sort_index()


def label_momentum(row):
    # Define momentum conditions
    if (
        row['own_scoring_streak'] >= 3 or
        row['last_5_play_points'] >= 7 or
        row['opponent_consecutive_misses'] >= 3 or
        row['fouls_drawn_last_5'] >= 2
    ):
        return 1
    return 0


if __name__ == "__main__":
    path = input("Enter path to your dataset CSV: ")
    df = pd.read_csv(path)

    print("Adding momentum features...")
    df = add_momentum_features(df)

    print("Labeling momentum...")
    df['momentum_label'] = df.apply(label_momentum, axis=1)

    output_path = path.replace(".csv", "_with_momentum.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Done. Saved to {output_path}")
