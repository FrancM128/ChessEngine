import {useState , useEffect} from 'react';
import {Chessboard} from 'react-chessboard';
import {Chess} from 'chess.js';


export default function ChessMatch(){
    const [game,setGame] = useState(new Chess());
    const [playerColor,setPlayerColor] = useState('white');
    
    const [isThinking , setIsThinking] = useState(false);
    const [isOver, setIsOver] = useState(false);

    useEffect(() => {
        if(playerColor === 'black' && game.history().length === 0){
            makeCPUMove(game.fen());
        }
    },[playerColor])

    function onDrop(sourceSquare, targetSquare) {
        
        if (isThinking)return false;
        if (game.turn() !== playerColor[0]) return false;
        
        const copyGame = new Chess(game.fen());
    
        try {
            const mossa = copyGame.move({
                from: sourceSquare,
                to: targetSquare,
                promotion: 'q',
            });
            
            if (mossa === null) return false;
            setGame(copyGame);
            
            if (copyGame.isGameOver()) {
                setTimeout(() => {
                    console.log(copyGame.pgn());
                    alert("Partita finita!");
                    setIsOver(true);
                }, 500);
                return true; 
            }

            setTimeout(() => makeCPUMove(copyGame.fen()), 250);
            return true;
        } catch (e) {
            console.error("Errore con la libreria js", e);
            return false;
        }
    }

     async function makeCPUMove(fenCurrent){
        //Init.
        setIsThinking(true);
        
        try {
            //link
            const response = await fetch('.../get_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fen: fenCurrent })
            });

            if (!response.ok) throw new Error("Errore dal server HTTP");
            
            const data = await response.json();
            
            const newGame = new Chess(fenCurrent);
            newGame.move(data.move);
            setGame(newGame);
            if (newGame.isGameOver()){
                setIsOver(true);
                setTimeout(() => {
                    console.log(newGame.pgn())
                },500)
                return true;
            }
        } catch (error) {
            console.error("Errore di connessione:", error);
        } finally {
            setIsThinking(false); 
        }
     }

     function resetGame(){
        const nuovaPartita = new Chess();
        setGame(nuovaPartita);
        setIsThinking(false);
        setIsOver(false);
        if(playerColor === 'black'){
            setTimeout(() => makeCPUMove(nuovaPartita.fen()),250);
        }
     }

     const chessboardOptions = {
        position: game.fen(),
        onPieceDrop: onDrop,
        boardOrientation: playerColor,
        boardWidth: 500,
        customDarkSquareStyle: {backgroundColor : '#2d3740'},
        customLightSquareStyle: {backgroundColor : '#bad3e8'},
        customBoardStyle: {
            borderRadius: '4px',
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
        }
       
     };
     return (
        <div style={{ width: '100%', maxWidth: '600px', margin: '50px auto', textAlign: 'center' }}>
            <h2>MotoreScacchi (Colab)</h2>
            <p style={{ color: isThinking ? 'red' : 'green', fontWeight: 'bold', fontSize: '18px' }}>
                {isThinking ? "La CPU sta pensando..." : isOver ? "Match finito!" : "È il tuo turno"}
            </p>
           
            <div style={{ marginBottom: '15px', display: 'flex', justifyContent: 'center', gap: '15px' }}>
                <button 
                    onClick={() => setPlayerColor(playerColor === 'white' ? 'black' : 'white')}
                    disabled={isThinking}
                    style={{ 
                        padding: '8px 16px', 
                        cursor: isThinking ? 'not-allowed' : 'pointer', 
                        borderRadius: '4px',
                        backgroundColor: '#0000ff27',
                        border: '1px solid #ffffffd0'
                    }}
                >
                    Gioca coi {playerColor === 'white' ? 'Neri' : 'Bianchi'}
                </button>
                <button 
                    onClick={resetGame}
                    disabled={isThinking}
                    style={{ 
                        padding: '8px 16px', 
                        cursor: isThinking ? 'not-allowed' : 'pointer', 
                        borderRadius: '4px',
                        backgroundColor: '#52ef44', 
                        color: 'white',
                        border: 'none',
                        fontWeight: 'bold'
                    }}
                >
                    Nuova Partita
                </button>

            </div>
            
            <Chessboard {...chessboardOptions} />
        </div>
     );
}