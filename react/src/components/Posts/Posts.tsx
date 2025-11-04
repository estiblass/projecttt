import React, { FC, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import EntryService from '../../services/entry.service';
import './Posts.css';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import { string } from 'yup';
import { PostFrom } from '../../types/PostFrom';
import { Container, Spinner } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Offcanvas from 'react-bootstrap/Offcanvas';
import InputGroup from 'react-bootstrap/InputGroup';
import { useDispatch, useSelector } from 'react-redux';
import { StoreType } from '../../redux/store';
interface PostsProps { }

const Posts: FC<PostsProps> = () => {
  debugger
  const [postList, setPostList] = useState<any[]>([])
  const [load, setLoad] = useState<boolean>(false)
  const location = useLocation();
  const userId = location.state
  useEffect(() => {
    EntryService.getPostToUser(userId).then((res) => {
      setPostList(res.data);
      setLoad(true);
    })
  }, [])
  console.log(postList)
  return <div><h1>{`The message list of user number:${userId}`}</h1>
    {!load ? <div><Spinner className='mySpinner' animation="border" variant="info" /></div> : ''}
    <div className='post col-sm-6' dir="ltr">
      <table className="table rtl">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Title</th>
            <th scope="col">Body</th>
          </tr>
        </thead>
        <tbody>
          {
            postList?.map((a, index) => {
              return <tr>
                <th scope="row">{index + 1}</th>
                <td>{a.title}</td>
                <td>{a.body}</td>
              </tr>
            })
          }
        </tbody>
      </table>
    </div>
  </div>
};

export default Posts;
