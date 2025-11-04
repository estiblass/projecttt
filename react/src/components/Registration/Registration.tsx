import React, { FC, useEffect, useRef, useState } from 'react';
import './Registration.css';
import { useFormik } from 'formik';
import { RegistrationForm } from '../../types/RegistrationForm';
import * as yup from 'yup'
import { useNavigate, useLocation } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import store, { StoreType} from '../../redux/store';
import {User} from '../../types/User'
import { NodeWorker } from 'inspector';

interface RegistrationProps { }

const  Registration: FC<RegistrationProps> = () =>  {
  const allStore = useSelector((store: StoreType) => store)
	const _disptach = useDispatch();
  const sendToServer = (value: RegistrationForm) => {
    alert('all the object send to server')
   
  }
  const myNav = useNavigate();
  const goToPage = (target:string) => {
    myNav(target)
  }
  const allFull=()=>{
    const allInputsFull = ((!(myForm.errors.email))&&(!(myForm.errors.userNumber))&&(!(myForm.errors.lastName))&&(!(myForm.errors.firstName)));
    goToPage(allInputsFull ? '/home' : '');
  } 
  const myForm = useFormik({
    initialValues: {firstName: '',
                   lastName: '',
                   email: '',
                   userNumber: '',
                  },
    onSubmit: (values) => {
            _disptach({ type: 'SET_USER', data: {id: myForm.values.userNumber,
            name: ` ${myForm.values.firstName}${''}${myForm.values.lastName}`,
            username: '',
            email: myForm.values.email,
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
            } as User })
            
          allFull()},
    
    validationSchema: yup.object().shape({
      firstName: yup.string().required('The field is required'),
      lastName: yup.string().required('The field is required'),
      userNumber:yup.string().required('The field is required'),
      email: yup.string().email('Invalid email').required('The field is required')
    }),
  })
  return <div>
      <React.Fragment>
        <form onSubmit={myForm.handleSubmit}>
          <h3>Login Here</h3>
          <label htmlFor="firstName">First name</label>
          <input type="text" placeholder="Enter a user first name" onBlur={myForm.handleChange} id="firstName" name="firstName"/>
          {myForm.errors.firstName ? <p className="error-p">* {myForm.errors.firstName}</p> : ''}
          <label htmlFor="last name">last name</label>
          <input type="text" placeholder="Enter a user last name" onBlur={myForm.handleChange} id="lastName" name="lastName" />
          {myForm.errors.lastName ? <p className="error-p">* {myForm.errors.lastName}</p> : ''}
          <label htmlFor="User number">User number</label>
          <input type="text" placeholder="Enter user number" onBlur={myForm.handleChange}  id="userNumber" name="userNumber" />
          {myForm.errors.userNumber ? <p className="error-p">* {myForm.errors.userNumber}</p> : ''}
          <label htmlFor="Email">Email</label>
          <input type="text" placeholder="Enter email" onBlur={myForm.handleChange} id="email" name="email" />
          {myForm.errors.email ? <p className="error-p">* {myForm.errors.email}</p> : ''}
          <button type="submit" >Log In</button>
          <div className="social">
            <div className="go"><i className="fab fa-google"></i>  Google</div>
            <div className="fb"><i className="fab fa-facebook"></i>  Facebook</div>
          </div>
        </form>
      </React.Fragment>
    </div>

}

export default Registration;
