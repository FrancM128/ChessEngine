import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import chess
from tqdm.notebook import tqdm

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 769 = 768 input + 1 per il padding ovvero i pezzi mancanti
        self.feature_layer = nn.Embedding(769, 256, padding_idx=768)
        torch.nn.init.normal_(self.feature_layer.weight, mean=0.0, std=0.01)
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)
        torch.nn.init.zeros_(self.output.weight)

    def forward(self, x):
        features = self.feature_layer(x)
        accumulator = features.sum(dim=1)
        x = self.activation(accumulator)
        x = self.layer1(x)
        x = self.activation(x)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.layer2(x)
        x = self.activation(x)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.layer3(x)
        x = self.activation(x)
        return self.output(x)

class NNUEEngine:
    def __init__(self, model_path="models/nnue.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet().to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded")
        except FileNotFoundError:
            print("model non trovato")

    def fen_to_tensor_input(self, fen: str) -> torch.Tensor:
        board = chess.Board(fen)
        indices = []
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece_map[piece.symbol()] * 64 + square
                indices.append(idx)
        padded = indices + [768] * (32 - len(indices))
        return torch.tensor([padded], dtype=torch.long)

    def evaluate_position(self, fen: str) -> float:
        tensor_in = self.fen_to_tensor_input(fen).to(self.device)
        with torch.no_grad():
            output = self.model(tensor_in)
        return output.item()

    def train_model(self, dataset_path: str, save_path: str, epochs: int = 20, batch_size: int = 2048):
        print(f"Inizio Training :\n Dispositivo in uso: {self.device.upper()}")
        try:
            inputs, targets = torch.load(dataset_path)
        except FileNotFoundError:
            print(f"Errore: File '{dataset_path}' non trovato.")
            return

        dataset = TensorDataset(inputs, targets)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        total_steps = epochs * (len(train_loader) + len(val_loader))
        training_bar = tqdm(total=total_steps, desc="Training Engine", unit="batch")
        best_val_mae = float('inf')

        for epoch in range(epochs):
            self.model.train()
            train_loss, total_mae = 0.0, 0.0

            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_targets_scaled = batch_targets / 15.0

                optimizer.zero_grad()
                predictions_scaled = self.model(batch_inputs)
                loss = criterion(predictions_scaled, batch_targets_scaled)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                with torch.no_grad():
                    predictions_real = predictions_scaled * 15.0
                    mae_val = torch.abs(predictions_real - batch_targets).mean().item()
                    total_mae += mae_val

                training_bar.update(1)
                training_bar.set_postfix({'Epoca': f"{epoch+1}/{epochs}", 'Loss': f"{loss.item():.4f}"})

            self.model.eval()
            val_mae = 0.0
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    predictions_real = self.model(batch_inputs) * 15.0
                    val_mae += torch.abs(predictions_real - batch_targets).mean().item()
                    training_bar.update(1)

            avg_val_mae = val_mae / len(val_loader)
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(self.model.state_dict(), save_path)
            
            print(f"Epoca [{epoch+1}/{epochs}] | Train MAE: {total_mae/len(train_loader):.2f} | Val MAE: {avg_val_mae:.2f}")

        training_bar.close()

if __name__ == "__main__":
    bot = NNUEEngine()
    bot.train_model(dataset_path="dataset.pt", save_path="models/nnue.pth", epochs=100, batch_size=2048)
