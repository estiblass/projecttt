import React from 'react';
import logo from './logo.svg';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import "bootstrap-icons/font/bootstrap-icons.css";
import Game from './components/Game/Game';
import Registration from './components/Registration/Registration';
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import Home from './components/Home/Home';
import About from './components/About/About';
import Manager from './components/Manager/Manager';
import User from './components/User/User';
import NotFound from './components/NotFound/NotFound';
import Posts from './components/Posts/Posts';
import { useDispatch, useSelector } from 'react-redux';
import { StoreType } from './redux/store';
import NewUser from './components/NewUser/NewUser';
import UploadFile from './components/UploadFile/UploadFile';


function App() {
  
 //האזנה לחנות
 const allStore=useSelector((store:StoreType)=>store)
 const _disptach=useDispatch();
 debugger
 console.log(allStore.userReducer.id)
 console.log(typeof(allStore.userReducer.id))
  return (
    <BrowserRouter>
      <Routes>
        {/* <Route path="/" element={<Registration></Registration>}></Route> */}
        <Route path="/" element={<UploadFile></UploadFile>}></Route>
        {/* <Route path="/home" element={<Home></Home>}>
          <Route path="" element={allStore.userReducer.id==="7" ? <Manager></Manager>:<User></User>}></Route>
          <Route path="about" element={<About></About>}></Route>
          <Route path="game" element={<Game></Game>} ></Route>
          <Route path="user/:userId/posts" element={<Posts></Posts>} ></Route>
          <Route path="newUser" element={<NewUser></NewUser>} ></Route>
          <Route path="manager" element={<Manager></Manager>} ></Route>
          <Route path="user" element={<User></User>} ></Route>
        </Route> */}
        <Route path="*" element={<NotFound></NotFound>}></Route>
      </Routes>
    </BrowserRouter>

  );
}

export default App;


