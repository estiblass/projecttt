import React, { FC, useEffect, useRef, useState } from 'react';
import './NewUser.css';
import { useDispatch, useSelector } from 'react-redux';
import { StoreType } from '../../redux/store';
import EntryService from '../../services/entry.service';
import { User } from '../../types/User';
import { Col, Form, Spinner } from 'react-bootstrap';
import Offcanvas from 'react-bootstrap/Offcanvas';
import Button from 'react-bootstrap/Button';
import InputGroup from 'react-bootstrap/InputGroup';
import axios from 'axios';
interface NewUserProps { }

const NewUser = () => {
  const [usersList, setUsersList] = useState<any[]>([])
  const [load, setLoad] = useState<boolean>(false)
  const refName = useRef<any>();
  const refId = useRef<any>();
  const refEmail = useRef<any>();
  const [newUser, setNewUser] = useState<User>({
    id: '',
    name: '',
    username: '',
    email: '',
    address: {
      street: '',
      suite: '',
      city: '',
      zipcode: '',
      geo: {
        lat: '',
        lng: '',
      },
    },
  });
  useEffect(() => {
    EntryService.getListUsers().then((res) => {
      console.log(res.status)
      setUsersList(res.data);
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
      addUser();
    }
  };

  const [show, setShow] = useState(false);

  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  const addUser = () => {
    let user={
      id: refId.current.value,
      name: refName.current.value,
      username: refName.current.value,
      email: refEmail.current.value,
      address: {
        street: '',
        suite: '',
        city: '',
        zipcode: '',
        geo: {
          lat: '',
          lng: '',
        },
      },
    }
    setNewUser(user);
    // axios
    //   .post('https://jsonplaceholder.typicode.com/users', newUser)
    //   .then((response) => {
    //     if (response.status === 201) {
          const newUserData = { ...user, id: usersList.length + 1 };
          setUsersList([...usersList, newUserData]);
          setNewUser({
            id: refId.current.value,
            name: refName.current.value,
            username: refName.current.value,
            email: refEmail.current.value,
            address: {
              street: '',
              suite: '',
              city: '',
              zipcode: '',
              geo: {
                lat: '',
                lng: ''
              }
            }
          });
          setShow(false);
      //   } else {
      //     console.error(`Failed to create user with status code ${response.status}`);
      //   }
      // })
      // .catch((error) => {
      //   console.error(error);
      // });
  };
  console.log(typeof (refId))
  return <div><div className='users' dir="ltr">
    <table className="table">
      <thead>
        <tr>
          <th scope="col">#</th>
          <th scope="col">Name</th>
          <th scope="col">Email</th>
          <th scope="col">Id</th>
        </tr>
      </thead>
      <tbody>
        {!load ? <div><Spinner className='mySpinner' animation="border" variant="info" /></div> : ''}
        {
          usersList?.map((a, index) => {
            return <tr><th scope="row">{index + 1}</th>
              <td>{a.username}</td>
              <td>{a.email}</td>
              <td>{a.id}</td>
            </tr>
          })
        }
      </tbody>
    </table>
  </div>
    <div className='addUser'>
      <Button variant="outline-info" className='buttonUser' onClick={handleShow}> Add user</Button>{' '}
      <Offcanvas show={show} onHide={handleClose} placement='start' className='Offcanvas'>
        <Offcanvas.Header closeButton>
          <Offcanvas.Title className='NewUser'>New user</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body>
          <div>
            <div>
              <Form noValidate validated={validated} onSubmit={handleSubmit} className='form'>
                <Form.Group as={Col} md="13" controlId="validationCustom04">
                  <Form.Label>Name</Form.Label>
                  <Form.Control type="text" placeholder="Name" required ref={refName} />
                  <Form.Control.Feedback type="invalid">
                    The field is required.
                  </Form.Control.Feedback>
                </Form.Group>
                <Form.Group as={Col} md="13" controlId="validationCustom04">
                  <Form.Label>Email</Form.Label>
                  <Form.Control type="text" placeholder="Email" required ref={refEmail} />
                  <Form.Control.Feedback type="invalid">
                    The field is required.
                  </Form.Control.Feedback>
                </Form.Group>
                <Form.Group as={Col} md="13" controlId="validationCustom04">
                  <Form.Label>Id</Form.Label>
                  <Form.Control type="text" placeholder="Id" required ref={refId} />
                  <Form.Control.Feedback type="invalid">
                    The field is required.
                  </Form.Control.Feedback>
                </Form.Group>
                <Button type="submit" variant="outline-dark" className='button' >Add</Button>{' '}
              </Form>
            </div>
          </div>
        </Offcanvas.Body>
      </Offcanvas>
    </div>
  </div>
};

export default NewUser;
