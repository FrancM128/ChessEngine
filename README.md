<h1 align="center">Chess Engine</h1>

<p align="center">
  <em>Un motore scacchistico basato su rete NNUE, sviluppato in Python, con interfaccia in html da girare in locale attraverso ReactJS.</em>
</p>

---

## Descrizione
**Chess Engine** è un motore scacchistico sperimentale che combina la generazione di mosse tradizionale con la valutazione delle posizioni affidata all'intelligenza artificiale. 

Il progetto è pensato per esplorare l'addestramento e l'integrazione di architetture neurali nel gioco degli scacchi, si cerca di consolidare le teorie del deep learning attraverso la tavola degli scacchi.

### Tecnologie e Librerie
* **Linguaggio:** Python (AI e Dati),ReactJS(FrontEnd)
* **Rappresentazione Scacchiera e Mosse:** [python-chess](https://python-chess.readthedocs.io/)
* **Deep Learning:** [PyTorch](https://pytorch.org/)
* **FrontEnd:** ReactJS , collegamento server flask - react con Ngrok

---

## Dataset
L'addestramento del modello si basa su 10 milione di posizioni di partite reali per catturare pattern posizionali e tattici di variazioni standard. 

I dati utilizzati per il training provengono dai data dump mensili open-source di **[Lichess.org Open Database](https://database.lichess.org/)** 
(Lichess January 2026)(rilasciati in pubblico dominio con licenza CC0). 

* **Filtraggio:** Sono state utilizzate partite che avessero un elo maggiore o uguale a 2000 elo, esclusivamente con Time Control di minimo 10 minuti per giocatore. Parametri facilmente cambiabili all'interno dell'ETL grazie al visitor di python-chess, utilizzando Python esportandolo attraverso un tensore di Pytorch. 
* * **Estrazione Features:** Le partite in formato PGN sono state processate per estrarre le singole posizioni (in formato FEN) e i relativi risultati finali o le valutazioni di Stockfish, preparando i tensori necessari per l'addestramento. 

---

##  Architettura NNUE 
Il cuore della valutazione di questo engine è una rete neurale ottimizzata, strutturata seguendo i principi delle architetture **NNUE** (Efficiently Updatable Neural Network). 

Invece di affidarsi a una complessa funzione di valutazione scritta a mano, la rete impara a stimare il vantaggio direttamente dai dati storici.

* **Input (Rappresentazione):** Rappresentato attraverso la Piece-Square representation
* * **Topologia della Rete:** La rete è composta da un primo strato di input di 768 (64 celle x 12 pezzi) con 3 hidden layer 256->128->64, attivazione tra i layers secondo ReLU
* **Output:** La rete restituisce uno scalare che rappresenta la valutazione statica della posizione dal punto di vista del giocatore che ha il tratto, dato in centipawns negativo o positivo se rispettivamente dal punto di vista del nero o bianco. Le posizioni iniziali sono scelte casualmente dal fine "gm2001.bin" attraverso polyglot in python
* **Integrazione:** L'inferenza avviene tramite i tensori di PyTorch, chiamata in fase di ricerca (Search) sulle foglie dell'albero delle varianti attraverso Negamax.

---

## 📄 Licenza
Questo software è rilasciato sotto licenza **GPLv3**. 
L'engine integra `python-chess` (GPLv3) per la logica scacchistica e `PyTorch` (BSD-3-Clause) per il calcolo neurale.
