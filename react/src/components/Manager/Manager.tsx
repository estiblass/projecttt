
import React, { FC, forwardRef, useEffect, useImperativeHandle, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import EntryService from '../../services/entry.service';
import './Manager.css';
import { DataGrid } from '@mui/x-data-grid';
import Posts from '../Posts/Posts';
import { Button, Spinner } from 'react-bootstrap';
import { useDispatch, useSelector } from 'react-redux';
import { StoreType } from '../../redux/store';


interface ManagerProps { }

const Manager = () => {
  debugger
  const [list, setList] = useState<any[]>([])
  const [load, setLoad] = useState<boolean>(false)
  const navigate = useNavigate();
  const allStore = useSelector((store: StoreType) => store)
  useEffect(() => {
    EntryService.getListUsers().then((res) => {
      console.log(res.status)
      setList(res.data);
      setLoad(true);
    })
  }, [])
  console.log(list);
  const displayPosts = (userId: string) => {
    navigate(`/home/user/${userId}/posts`, { state: userId })
  }
  const [sortColumn, setSortColumn] = useState("username");
  const [sort, setSort] = useState(false);
  
  const sortIt = () => {
    setList([...list].sort((a, b) => a.username.localeCompare(b.username)));
  }
  return <div><h3>{`hello${allStore.userReducer.name}`}</h3>
    <table className="table tableManager">
      <thead>
        <tr>
          <th scope="col">#</th>
          <th scope="col">Name</th>
          <th scope="col">Email</th>
          <th scope="col">Id</th>
        </tr>
      </thead>
      <tbody>
        {!load ? <div><Spinner className='mySpinner' animation="border" variant="info " /></div> : ''}
        {
          list?.map((a, index) => {
            return <tr onClick={() => { displayPosts(a.id) }}>
              <th scope="row">{index + 1}</th>
              <td>{a.username}</td>
              <td>{a.email}</td>
              <td>{a.id}</td>
            </tr>
          })
        }
      </tbody>
    </table>
    <Button variant="outline-info" className='button' onClick={() => { sortIt() }}>Sort list</Button>{' '}
  </div>
}


export default Manager;
