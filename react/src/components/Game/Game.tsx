import React, { FC , useRef, useState} from 'react';
import './Game.css';
import { Xo } from '../../types/xo.type';
import Button from 'react-bootstrap/Button';


interface GameProps {}
const Game: FC<GameProps> = () => {
// const Game = () => {
  const [matrix,setMatrix]=useState([
    [{} as Xo,{} as Xo,{} as Xo],
    [{} as Xo,{} as Xo,{} as Xo],
    [{} as Xo,{} as Xo,{} as Xo]
  ])
  const [turn,setTurn]=useState('X');
  const [winnerX,setWinnerX]=useState(false);
  const [winnerO,setWinnerO]=useState(false);
  const[newGame,setnewGame]=useState(true);
  const[count,setCount]=useState(0);


  const turnX=(row:number,col:number)=>{
    if(matrix[row][col].type==null){ //כדי שלא ילחצו פעמיים על אותו משבצת ואז התוכן יוכל להשתנות 
    matrix[row][col].type='X'
    matrix[row][col].pressed=true
    setCount(count+1)
    setTurn('O')
    if(matrix[0][0].type==='X'&&matrix[0][1].type==='X'&&matrix[0][2].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[1][0].type==='X'&&matrix[1][1].type==='X'&&matrix[1][2].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[2][0].type==='X'&&matrix[2][1].type==='X'&&matrix[2][2].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[0][0].type==='X' && matrix[1][0].type==='X' && matrix[2][0].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[0][1].type==='X'&&matrix[1][1].type==='X'&&matrix[2][1].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[0][2].type==='X'&&matrix[1][2].type==='X'&&matrix[2][2].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[0][0].type==='X'&&matrix[1][1].type==='X'&&matrix[2][2].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
    if(matrix[0][2].type==='X'&&matrix[1][1].type==='X'&&matrix[2][0].type==='X')
    {
      setWinnerX(true)
      setnewGame(false);
    }
  }
  }


  const turnO=(row:number,col:number)=>{
    if(matrix[row][col].type==null){
    matrix[row][col].type='O'
    matrix[row][col].pressed=true
    setCount(count+1)
    setTurn('X')
    if(matrix[0][0].type==='O'&&matrix[0][1].type==='O'&&matrix[0][2].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[1][0].type==='O'&&matrix[1][1].type==='O'&&matrix[1][2].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[2][0].type==='O'&&matrix[2][1].type==='O'&&matrix[2][2].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[0][0].type==='O'&&matrix[1][0].type==='O'&&matrix[2][0].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[0][1].type==='O'&&matrix[1][1].type==='O'&&matrix[2][1].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[0][2].type==='O'&&matrix[1][2].type==='O'&&matrix[2][2].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[0][0].type==='O' && matrix[1][1].type==='O' && matrix[2][2].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
    if(matrix[0][2].type==='O'&&matrix[1][1].type==='O' && matrix[2][0].type==='O')
    {
      setWinnerO(true)
      setnewGame(false);
    }
  }
  }

  const takingTurns =(row:number,col:number)=>{
    if(!winnerO && !winnerX && count<9){
    turn==='X' ? turnX(row,col) : turnO(row,col);
  }
   
  }
  

 return <div className='game'>
   {
    <div>
      {newGame?<div className="startGame">Let's play!!!</div>:' '}
    </div>
  }
  {
   <div>
   {winnerO?<div className='winner'>The winner is O </div>:''}
   {winnerX?<div className='winner'>The winner is X </div>:''}
   {count===9 && !winnerO && !winnerX ?<div className="startGame">No one won :( </div>:''}
   </div>
  }
 {
   matrix.map((row,index)=>{
     return <div className="gameBoard">
       {row.map((col,infexCol)=>{
        debugger
        return<Button variant="danger" onClick={() => takingTurns(index,infexCol)}>{matrix[index][infexCol].pressed?matrix[index][infexCol].type:' ' }</Button>     })}
     </div>
   })
 }
 </div>

}

export default Game;


