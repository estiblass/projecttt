import React, { FC, useState } from 'react';
import { Link, Outlet, useNavigate, useLocation } from 'react-router-dom';
import './Home.css';
import { useSelector,useDispatch } from 'react-redux';
import { StoreType } from '../../redux/store';

interface HomeProps { }
const Home: FC<HomeProps> = () => {
  const navigate = useNavigate();
  const allStore = useSelector((store: StoreType) => store)
	const _disptach = useDispatch();
 
  return (<div>
    <nav className="navbar navbar-expand-lg navbar-light bg-light">
      <div className="collapse navbar-collapse" id="navbarNavDropdown">
        <ul className="navbar-nav">
          <li className="nav-item">
            <a onClick={() => { navigate(`/home/about`) }} className="nav-link" href="#">About</a>
          </li>
          <li className="nav-item">
            <a onClick={() => { navigate(`/home/game`) }} className="nav-link" href="#">Game</a>
          </li>
          <li className="nav-item">
            <a onClick={() => { navigate(`/home/newUser`) }} className="nav-link" href="#">New user</a>
          </li>
          <li className="nav-item back">
            <a onClick={() => { {if(allStore.userReducer.id===''){navigate(`*`)}else{navigate(allStore.userReducer.id==='7'?`/home/manager`:`/home/user`)}}}}className="nav-link" href="#">Back to my page</a>
          </li>
        </ul>
      </div>
    </nav>
    <Outlet></Outlet>
  </div>
  )
}


export default Home;


