import React, { FC } from 'react';
import './NotFound.css';
import { Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

interface NotFoundProps {}

const NotFound: FC<NotFoundProps> = () => {
  const navigate = useNavigate();
//   <div className="NotFound">
//  <h1>Not found.</h1>
//  <h2>You can try to register first and then log in.</h2>
//  <Button onClick={()=>{navigate(`/`)}} variant="danger" className='bt' >Log in</Button>
// </div>
return(  <div className="NotFound">
<h1>Not found.</h1>
<h2>You can try to register first and then log in.</h2>
<Button onClick={()=>{navigate(`/`)}} variant="danger" className='bt' >Register here</Button>
</div>)
};

export default NotFound;
