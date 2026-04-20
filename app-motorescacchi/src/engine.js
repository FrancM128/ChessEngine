
import {Chess} from 'chess.js'
import {getBookMoves} from 'polyglot-book-js'
import * as ort from 'onnxruntime-web'


ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

class Engine{
    /*2 parts of engine : 
    -take random move from the opening book (gm2001.bin, ... )
    -search in negamax (<10s) using a NNUE as evaluon function
    */
        constructor() {
            this.nnue = null;
            this.polyglotBuffer = null;
            this.isAborted = false;
            //chess pieces counted from 0 to 5 white, 6 to 11 black
            this.pieceMap= {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
            }
            
        }
       async init(modelPath, bookPath){
        try{
           this.nnue = await ort.InferenceSession.create(modelPath);
           const response = await fetch(bookPath);
           this.polyglotBuffer = await response.arrayBuffer();
            console.log("NNUE e Libro di aperture caricati");
        }catch (e){
            console.error("Errore durante l'inizializzazione" , e);
        }
       }
        //Opening book response, zobrist file         //function to convert the FEN board to a tensor 769x1 (12 pieces * 64squares) + 1padding using pieceMap(int32)  
        //FEN : it has 8 parts divided by '/' representing raws starting from the 8th ,
        //  the int32 is the n. of squares empty, as pieceMap  Uppercase for white , first letter as piece
        fenToTensorInput(fen){
            const board = new Chess(fen);
            const indices = new Int32Array(32);
            indices.fill(768);
            let count = 0;
            for(let r = 0 ; r< 8 ; r++){
                for(let c = 0 ; c<8; c++){
                    const square = String.fromCharCode(97 + c) + (r+1);
                    const piece = board.get(square);
                    if(piece && count < 32) {
                        const symbol = this.piece.color === 'w' ? piece.type.toUpperCase() : piece.type.toLowerCase();
                        const squareIdx = r*8 + c;
                        const idx = this.pieceMap[symbol] * 64 + squareIdx;
                        indices[count] = idx;
                        count++;
                    }
                }
            }

            return new ort.Tensor('int32',indices,[1,32]);
        }
        /** 
         * @param {string} fen 
         * @returns float32
         */
        async evaluate(fen) {
            if(!this.nnue) return 0;
            const feeds = {input: this.fenToTensorInput(fen)};
            const results = await this.nnue.run(feeds);

            const outputName = this.nnue.outputNames[0];
            const score = results[outputName].data[0];

            return (score);
        }

        //Search Algorithm
        //ordiring moves: includes takes -> just from-to square
        async orderMoves(board){
            return board.moves().sort((a,b) => { return b.includes('x') - a.includes('x')});
        }


        /*Negamax
        -give the score positive whoever moves
        -logic: simple minimax with ab pruning, swapping them every interaction and invert score sign make it different
        */
        async negamax(board,depth,alpha,beta,startTime,maxTimeMs){
            //time check
            if(Date.now - startTime > maxTimeMs){ 
                this.isAborted = true;
                return [0,null];
            }
            
            let score = 0;
            //Base case
            if (board.isCheckmate()) return [-Infinity + depth,null];
            if (board.isGameOver()) return [0, null];
            if (depth == 0){
                score = this.evaluate(board.fen())
                const molt = board.turn() == 'w' ?  1 : -1;
                return [score * molt,null];
            }
            

            //Recursive Case
            let best_move = null;
            let max_score = -Infinity;
            const oMoves = this.orderMoves(board);

            for(let i = 0; i< oMoves.length;i++){
                board.move(oMoves[i]);
                let result = await this.negamax(board,depth-1,-beta,-alpha)[0];
                let score = -result[0];
                board.undo();
                if (score > max_score){
                    max_score = score;
                    best_move = oMoves[i];
                }
                alpha = Math.max(alpha,max_score);
                if (alpha >= beta) break;
            }
            return [max_score , best_move];
        }

        //call negamax or watch opening book, upper lever to simplify and time management
        async searchBestMove(board,MaxTimeMs){
            if(this.polyglotBuffer){
                const bookMoves = getBookMoves(board.fen(),this.polyglotBuffer);
                if(bookMoves.length > 0){
                    const bookMove = bookMoves[0].uci;
                    return [bookMove,true];
                }
            }
            console.log("no book move found, start negamax")
            //negamax
            const startTime = Date.now();
            let bestMove = null;
            let found = false;
            let currentDepth = 1;
            while(Date.now() - startTime < MaxTimeMs){
                const result = this.negamax(board,currentDepth,-Infinity,Infinity,startTime,MaxTimeMs);
                if (this.isAborted) break;
                if(Date.now() - startTime <= MaxTimeMs && result[1] !== null){
                    bestMove = result[1];
                    found = true;
                }
                currentDepth++ ;
                if(currentDepth > 20) break;
            }
            
            return [bestMove,found];
            
        }

        
        
    }
export default Engine;