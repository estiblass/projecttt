import React, { FC, forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import './User.css';
import EntryService from '../../services/entry.service';
import { useNavigate, useParams } from 'react-router-dom';
import Form from 'react-bootstrap/Form';
import { string } from 'yup';
import { PostFrom } from '../../types/PostFrom';
import Button from 'react-bootstrap/Button';
import InputGroup from 'react-bootstrap/InputGroup';
import { Container, Spinner } from 'react-bootstrap';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Offcanvas from 'react-bootstrap/Offcanvas';
import { useDispatch, useSelector } from 'react-redux';
import { StoreType } from '../../redux/store';
interface UserProps { }
const User = () => {
  const [postList, setPostList] = useState<any[]>([])
  const [load, setLoad] = useState<boolean>(false)
  const refTitleInput = useRef<any>();
  const refBodyInput = useRef<any>();
  const allStore = useSelector((store: StoreType) => store)
  const _disptach = useDispatch();
  const userId = allStore.userReducer.id
  const [newPost, setNewPost] = useState<PostFrom>({
    userId: '',
    id: '',
    title: '',
    body: ''
  });
  console.log(typeof (allStore.userReducer.id))
  useEffect(() => {
    EntryService.getPostToUser(userId).then((res) => {
      console.log(res.status)
      setPostList(res.data);
      setLoad(true);
    })
  }, [])

  const [validated, setValidated] = useState(false);
  const handleSubmit = (event: any) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (form.checkValidity() === false) {
      event.stopPropagation();
    } else {
      setValidated(true);
      addPost();
    }
  };
  const [show, setShow] = useState(false);

  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  const addPost = () => {
    debugger
    let post = {
      userId: allStore.userReducer.id,
      id: allStore.userReducer.id,
      title: refTitleInput.current.value,
      body: refBodyInput.current.value
    }
    setNewPost(post)
    const newPostData = { ...post, id: postList.length + 1 };
    let arr=[...postList, newPostData]
    setPostList(arr);
    setNewPost({
      userId: allStore.userReducer.id,
      id: allStore.userReducer.id,
      title: refTitleInput.current.value,
      body: refBodyInput.current.value
    });
    setShow(false);
  };

  return <div><h3>{`hello${allStore.userReducer.name}`}</h3><div className='post col-sm-6' dir="ltr">
    <table className="table">
      <thead>
        <tr>
          <th scope="col">#</th>
          <th scope="col">Title</th>
          <th scope="col">Body</th>
        </tr>
      </thead>
      <tbody>
        {!load ? <div><Spinner className='mySpinner' animation="border" variant="info" /></div> : ''}
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
    <div className='addMessage'>
      <Button variant="outline-info" className='buttonPost' onClick={handleShow}> Add post</Button>{' '}
      <Offcanvas show={show} onHide={handleClose} placement='end' className='Offcanvas'>
        <Offcanvas.Header closeButton>
          <Offcanvas.Title className='NewPost'>New post</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body>
          <div>
            <Form noValidate validated={validated} onSubmit={handleSubmit} className='form'>
              <Form.Group as={Col} md="13" controlId="validationCustom04">
                <Form.Label>Title</Form.Label>
                <Form.Control type="text" placeholder="Title" required ref={refTitleInput} />
                <Form.Control.Feedback type="invalid">
                  The field is required.
                </Form.Control.Feedback>
              </Form.Group>
              <Form.Group as={Col} md="13" controlId="validationCustom04">
                <Form.Label>Body</Form.Label>
                <Form.Control type="text" placeholder="Body" required ref={refBodyInput} />
                <Form.Control.Feedback type="invalid">
                  The field is required.
                </Form.Control.Feedback>
              </Form.Group>
              <Button type="submit" variant="outline-dark" className='button' >Add</Button>{' '}
            </Form>
          </div>
        </Offcanvas.Body>
      </Offcanvas>
    </div>
  </div>
};

export default User;



