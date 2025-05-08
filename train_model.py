from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any
import polars as pl
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import DataCreationConfig, TrainingConfig

pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(20)


class OptionPriceDataset(Dataset):
    def __init__(self, dataframe: pl.DataFrame, features_cols: tuple[str], target_cols: tuple[str]):
        self.features = torch.tensor(dataframe.select(features_cols).to_numpy(), dtype=torch.float32).to(TrainingDevice.TRAINING_DEVICE)
        self.targets = torch.tensor(dataframe.select(target_cols).to_numpy(), dtype=torch.float32).to(TrainingDevice.TRAINING_DEVICE)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return self.features[idx], self.targets[idx]


class OptionPricePredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super(OptionPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def prepare_data(df: pl.DataFrame, features: tuple[str], target: tuple[str]) -> tuple[DataLoader[Any], DataLoader[Any]]:

    # filter out invalid/null Data
    filtered_df = df.filter(
        pl.col('h').is_not_null() &
        pl.col('l').is_not_null() &
        pl.col('o').is_not_null() &
        pl.col('c').is_not_null() &
        pl.col('time_to_expiry_days').is_not_null() &
        pl.col('callput').is_not_null() &
        pl.col('strike').is_not_null() &
        pl.col('risk_free_rate').is_not_null() &
        pl.col('ul_h').is_not_null() &
        pl.col('ul_l').is_not_null() &
        pl.col('ul_o').is_not_null() &
        pl.col('ul_c').is_not_null()
    )

    filtered_df = filtered_df.with_columns(pl.when(pl.col('callput') == 'C').then(1).otherwise(0).alias('callput'))

    random_state: int = 42
    train_df, val_df = train_test_split(filtered_df, test_size=TrainingConfig.VAL_SIZE, random_state=random_state)

    train_dataset = OptionPriceDataset(train_df, features, target)
    val_dataset = OptionPriceDataset(val_df, features, target)

    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.VAL_BATCH_SIZE)

    return train_loader, val_loader


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer):

    for epoch in range(TrainingConfig.TRAINING_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:

            inputs = inputs.to(TrainingDevice.TRAINING_DEVICE)
            targets = targets.to(TrainingDevice.TRAINING_DEVICE)

            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_loss += loss.item() * inputs.size(0)

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculations during validation
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        tprint(f'Epoch {epoch + 1}/{TrainingConfig.TRAINING_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


def get_all_tables_paths(dir_path: str) -> list[str]:

    if not os.path.exists(dir_path):
        raise ValueError(f'Directory {dir_path} does not exist.')

    table_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.arrow'):
                if not any(map(lambda x: x in root, DataCreationConfig.TRAIN_IGNORE_FOLDERS.split(','))):
                    table_paths.append(os.path.join(root, file))

    return table_paths


def tprint(*args):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : {' '.join(args)}")


@dataclass
class TrainingDevice:
    TRAINING_Device: torch.device


def startup():
    table_paths: list[str] = get_all_tables_paths(TrainingConfig.TRAINING_DATA_PATH)

    features: tuple[str] = TrainingConfig.FEATURES
    target: tuple[str] = TrainingConfig.TARGET

    TrainingDevice.TRAINING_DEVICE = torch.device('cpu')
    if TrainingConfig.TRAINING_HARDWARE == 'CUDA':
        if torch.cuda.is_available():
            TrainingDevice.TRAINING_DEVICE = torch.device('cuda')
            tprint('Using GPU for training.')
        else:
            tprint('No GPU found, using CPU instead.')

    model = OptionPricePredictor(input_size=len(features), output_size=len(target)).to(TrainingDevice.TRAINING_DEVICE)

    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for i, table_path in enumerate(table_paths, start=1):
        tprint(f'Processing table {i:>5} / {len(table_paths):,}{i / len(table_paths) * 100:>7.2f}%: {table_path}')
        df = pl.read_ipc(table_path, memory_map=False)

        train_loader, val_loader = prepare_data(df, features, target)

        train_model(model, train_loader, val_loader, criterion, optimizer)

        if i % 50 == 0:
            FULL_PATH = os.path.join(os.path.dirname(__file__), TrainingConfig.MODEL_SAVE_NAME)
            torch.save(model, FULL_PATH)
            tprint(f'Model saved to {FULL_PATH}')


