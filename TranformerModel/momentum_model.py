import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Momentum Transformer Model
# ------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MomentumTransformer(nn.Module):
    categorical_columns = [
        'home', 'away', 'half', 'description', 'action_team', 'scoring_play', 'foul',
        'shot_team', 'shot_outcome', 'shooter', 'three_pt', 'free_throw', 'possession_before', 'possession_after'
    ]

    numerical_columns = [
        'game_id', 'play_id', 'secs_remaining', 'home_score', 'away_score', 'score_diff', 'play_length'
    ]

    def __init__(self, category_sizes, num_numeric, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            (col if col != 'half' else 'half_'): nn.Embedding(size, d_model) for col, size in category_sizes.items()
        })
        self.embedding_keys = [col if col != 'half' else 'half_' for col in category_sizes.keys()]
        self.numeric_proj = nn.Linear(num_numeric, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)

        # Load category encoders and scaler
        with open("category_encoders.pkl", "rb") as f:
            self.category_encoders = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

    def forward(self, x_cat, x_num):
        x_cat = x_cat.to(next(self.parameters()).device)
        x_num = x_num.to(next(self.parameters()).device)
        emb = sum(self.embeddings[col](x_cat[:, :, i]) for i, col in enumerate(self.embedding_keys))
        num_proj = self.numeric_proj(x_num)
        x = emb + num_proj
        x = self.pos_encoder(x)
        attention_scores = x.detach().mean(dim=2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output_layer(x), attention_scores

    def process_dataframe(self, df, sequence_length=5, make_labels=False, run_amount=5):
        df_encoded = df.copy()

        # Encode categorical columns
        for col in self.categorical_columns:
            df_encoded[col] = df_encoded[col].map(self.category_encoders[col]).fillna(
                self.category_encoders[col]['MISSING']).astype(int)

        X_cat, X_num, y = [], [], []
        max_i = len(df_encoded) - (sequence_length + (1 if make_labels else 0))

        for i in range(max_i):
            window = df_encoded.iloc[i:i + sequence_length]
            X_cat.append(window[self.categorical_columns].values)
            X_num.append(window[self.numerical_columns].values)

            if make_labels:
                next_diff = df_encoded.iloc[i + sequence_length]["points_next_min_diff"]
                label = 1 if abs(next_diff) >= run_amount else 0
                y.append(label)

        X_cat = np.array(X_cat)
        X_num = np.array(X_num)

        X_num = self.scaler.transform(X_num.reshape(-1, X_num.shape[-1])).reshape(X_num.shape)

        if make_labels:
            return torch.tensor(X_cat, dtype=torch.long), torch.tensor(X_num, dtype=torch.float32), torch.tensor(y,
                                                                                                                 dtype=torch.long)

        return torch.tensor(X_cat, dtype=torch.long), torch.tensor(X_num, dtype=torch.float32)


# ------------------------------
# Load model utility
# ------------------------------

def load_model(path="momentum_transformer.pth", category_sizes=None, num_numeric=len(MomentumTransformer.numerical_columns)):
    if category_sizes is None:
        # Load category_sizes from saved pickle
        with open("category_sizes.pkl", "rb") as f:
            category_sizes = pickle.load(f)

    model = MomentumTransformer(category_sizes, num_numeric)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
