
# %% [markdown]
# <h1>AI</h1>

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,random_split
import chess
import chess.polyglot
from tqdm.notebook import tqdm

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        #769 = 768 input + 1 per il padding ovvero i pezzi mancanti
        self.feature_layer = nn.Embedding(769, 256,padding_idx=768)
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
        x = nn.functional.dropout(x, p=0.2, training=self.training) # Fondamentale
        x = self.layer3(x)
        x = self.activation(x)
        return self.output(x)

class NNUEEngine:
    def __init__(self,model_path = "models/nnue.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet().to(self.device)

        try:
            self.model.load_state_dict(torch.load("models/nnue.pth",map_location=self.device))
            self.model.eval()
            print("Model loaded - ","cuda" if torch.cuda.is_available() else "cpu")
        except FileNotFoundError:
            print("model non trovato")

    def fen_to_tensor_input(self,fen:str) -> torch.Tensor:
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

    def evaluate_position(self, fen:str) -> float:
        tensor_in = self.fen_to_tensor_input(fen).to(self.device)
        with torch.no_grad():
            output = self.model(tensor_in)
        return output.item()
    def orderMoves(self, board : chess.Board)->list :
        moves = list(board.legal_moves)
        moves.sort(key = lambda m : board.is_capture(m),reverse = True)
        return moves

    def negamaxAlphabeta(self,board : chess.Board,depth : int,alpha : float,beta : float) -> tuple[float, chess.Move | None]:
        if board.is_checkmate():
            return -float('inf') + depth, None

        if board.is_game_over():
            return 0 , None

        if depth == 0 :
            score  = self.evaluate_position(board.fen())
            molt = 1 if board.turn == chess.WHITE else -1
            return molt * score, None

        best_move = None
        max_score = -float('inf')
        oMoves = self.orderMoves(board)

        for move in oMoves:
            board.push(move)

            score, _  = self.negamaxAlphabeta(board, depth-1, -beta, -alpha)
            score = -score
            board.pop()
            if score > max_score :
                max_score = score
                best_move = move
            alpha = max(alpha, max_score)
            if alpha >= beta:
                break

        return max_score,best_move


    def OpeningBookMove(self, board : chess.Board,book_path = "aperture/gm2001.bin"):
        
        try : 
            with chess.polyglot.open_reader(book_path) as reader:
                entry = reader.choice(board)
                return entry.move.uci()
        except IndexError:
            return None
        except FileNotFoundError:
            print("Libro non trovato")
            return None
                


    def search_best_move(self,fen : str,depth : int)-> chess.Move | None:
        board = chess.Board(fen)
        move = self.OpeningBookMove(board)
        
        if move == None:
            score,move = self.negamaxAlphabeta(board,depth,-float("inf"),float("inf"))
            print(f"Eval : {score:.4f} centipawns, Mossa : {move}")
        else:
            print("trovata nel libro")
        return move

    #training

    def train_model(self,dataset_path: str, save_path: str, epochs: int = 20, batch_size: int = 2048):

        print(f"Inizio Training :\n Dispositivo in uso: {self.device.upper()}")
        print(f"Caricamento dataset da '{dataset_path}' in memoria...")
        try:
            inputs, targets = torch.load(dataset_path)
        except FileNotFoundError:
            print(f"Errore: File '{dataset_path}' non trovato. Esegui prima l'ETL.")
            return


        dataset = TensorDataset(inputs, targets)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        print(f"Inizio addestramento ({epochs} epoche, Batch Size: {batch_size})...")


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


            train_loss = 0.0 # MSE
            total_mae = 0.0  # Mean Absolute Error

            for batch_inputs, batch_targets in train_loader:

                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Ottimizzazione
                batch_targets_scaled = batch_targets / 15.0

                optimizer.zero_grad()
                predictions_scaled = self.model(batch_inputs)
                loss = criterion(predictions_scaled, batch_targets_scaled)

                loss.backward()
                optimizer.step()


                train_loss += loss.item()

                # Monitoring
                with torch.no_grad():
                    predictions_real = predictions_scaled * 15.0
                    errore_assoluto = torch.abs(predictions_real - batch_targets)
                    mae_val = errore_assoluto.mean().item()
                    total_mae += mae_val

                training_bar.update(1)
                training_bar.set_postfix({
                    'Epoca': f"{epoch+1}/{epochs}",
                    'Loss (live)': f"{loss.item():.4f}",
                    'MAE (live)': f"{mae_val:.2f}"
                })

            self.model.eval()
            val_mae = 0.0

            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    predictions_scaled = self.model(batch_inputs)
                    predictions_real = predictions_scaled * 15.0
                    abs_error = torch.abs(predictions_real - batch_targets)
                    val_mae += abs_error.mean().item()

                    training_bar.update(1)
                    training_bar.set_postfix({'Epoca': f"{epoch+1}/{epochs}", 'Fase': 'Val'})

            # Calcolo corretto delle medie finali
            avg_loss = train_loss / len(train_loader)
            avg_mae = total_mae / len(train_loader)
            avg_val_mae = val_mae / len(val_loader)
            if avg_val_mae < best_val_mae:
                print(f"  --> Nuovo record! Val MAE sceso da {best_val_mae:.2f} a {avg_val_mae:.2f}. Salvataggio pesi...")
                best_val_mae = avg_val_mae
                # Sovrascrive il file solo se la rete è effettivamente migliorata nella generalizzazione
                torch.save(self.model.state_dict(), save_path)
            else:
                print(f"  --> Overfitting rilevato. Il record rimane {best_val_mae:.2f}.")

            print(f"Epoca [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.5f} | Train MAE: {avg_mae:.2f} | Val MAE: {avg_val_mae:.2f}")


        training_bar.close()

        print(f"\nTraining completato con successo. Pesi salvati in '{save_path}'.")


# %% [markdown]
# <h1>Server</h1>

# %%
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import time

NGROK_TOKEN = "...."
ngrok.set_auth_token(NGROK_TOKEN)   

bot = NNUEEngine()

app = Flask(__name__)
CORS(app) 

@app.route('/get_move', methods=['POST'])
def get_move():
    data = request.json
    fen = data.get('fen')
    
    print(f"Ricevuta richiesta per FEN: {fen}")
    
    
    best_move = bot.search_best_move(fen,4)
    time.sleep(30) 
    
    
    return jsonify({"move": best_move})


public_url = ngrok.connect(5000).public_url
print("\n" + "="*50)
print(f"{public_url}/get_move")
print("="*50 + "\n")


app.run(port=5000)


