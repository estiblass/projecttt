import axios from 'axios';
import React, { FC, useEffect, useRef, useState } from 'react';
import { Button, FormControl, InputGroup, Spinner } from 'react-bootstrap';
import { useSelector } from 'react-redux';
import { StoreType } from '../../redux/store';
import './UploadFile.css';
interface UploadFileProps { }

const UploadFile = () => {
  const inpFileRef = useRef<any>();
  const [listFile, setListFile] = useState<any[]>([])
  const [spinner, setSpinner] = useState<boolean>(false)
  const [file, setFile] = useState<File | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [data, setData] = useState(null);
  const [resalt, setResalt] = useState(null);
  const handleFileChange = (event: any) => {
    const uploadedFile = event.target.files[0];
    console.log(event.target.files[0])
    setSelectedFile(event.target.files[0]);
    setFile(uploadedFile);
  };

  //הצגת הקבצים שהועלו
  // const handleFileChange = (event: any) => {
  //   debugger

  //   setFile(uploadedFile);
  // };

  // const handleSubmit = async (event: any) => {
  //   event.preventDefault();
  //   const formData = new FormData();
  //   listFile.forEach((file, index) => {
  //     formData.append(`file${index}`, file);
  //   });
  //   console.log(formData)
  //   try {
  //     axios.post("http://127.0.0.1:5000/api/form", formData)
  //       .then(response => {
  //         console.log(response.data);
  //       })
  //       .catch(error => {
  //         console.error(error);
  //       });
  //   } catch (error) {
  //     console.error(error);
  //   }

  //   try {
  //     const response = await axios.get('http://127.0.0.1:5000/api/get');
  //     console.log(response.data.message);
  //   } catch (error) {
  //     console.error(error);
  //   }
  // }
  const handleSubmit = async (event: any) => {
    event.preventDefault();

    if (!selectedFile) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    if (selectedFile !== null) {
      formData.append('image', selectedFile);
    }

    try {
      await axios.post('http://127.0.0.1:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('Image uploaded successfully');
    } catch (error) {
      console.error('Error uploading image', error);
    }
    setSpinner(true)
    try {
      axios.get('http://127.0.0.1:5000/api/upload')
        .then(response => {
          // Handle the response from the Python server
          const result = response.data.result;
          setResalt(result)
          setSpinner(false)
          console.log(result)
          // Update the UI or perform any necessary actions with the result
        })
    }
    catch (error) {
      console.error('Error uploading image', error);
    }
  };
  const openDialog = (event: any) => {
    //current  -מאפיין יחיד הקיים להצביע, הוא מכיל את כל הערך של מי שהוא משוייך אליו
    inpFileRef.current.click();
  }

  const selectedFiles = (e: any) => {
    //e.target.files הסוג שלו הוא fileList ולא ניתן לעבור עליו בלולאה
    //המרת רשימת הקבצים למערך
    // איפוס המערך ודחיפת רשימת הקבצים
    // setListFile([... e.target.files]);
    //חיבור רשימת הקבצים שנטענו לפני יחד עם הקבצים שנטענו עכשיו
    let a = listFile.concat([...e.target.files]);
    setListFile([...a])
    handleFileChange(e)
  }

  // const sendFiles = () => {
  //   //אובייקט המאגד רישמה של קבצים לשליחה לשרת
  //   let data = new FormData();
  //   listFile.forEach((file, index) => {
  //     data.append(`file${index}`, file, file.name);
  //   })

  //   //טעינת קובצים לשרת נדרש לשלוח
  //   //נתיב+אובייקט של קבצים+ לעדכן את כותרת הבקשה ששולחים אובייקט של קבצים ולא אובייקט גייסון
  //   axios.post('/api/...', data, {
  //     headers: {
  //       'content-type': 'multipart/form-data'
  //     }
  //   })
  // }

  const dragFile = (event: any) => {
    event.preventDefault();
    debugger
  }


  return (
    <div dir="rtl" className="UploadFile">
      <div>
        <div className="title">
          <h1> המרת כתב יד לכתב מחשב</h1>
        </div>
        {/* <div onDragLeave={dragFile} onDrag={dragFile} onClick={openDialog} className="uploadFileC">click for upload yfile</div> */}
        <div>
          <input className="inputFile" onChange={selectedFiles} id="inpFile" ref={inpFileRef} type="file"></input>
        </div>
        <div>
          {file && (
            <div>
              <img className="showFile" src={URL.createObjectURL(file)} alt="Uploaded file" />
            </div>
          )}
        </div>
        <Button variant="secondary" onClick={handleSubmit} className="butt">send file</Button>{' '}
        {spinner ? (
          <Button variant="secondary" disabled>
            <Spinner
              as="span"
              animation="grow"
              size="sm"
              role="status"
              aria-hidden="true"
            />
            ...Processes the data
          </Button>
        ) : (
          ''
        )}
        {resalt && (
          <div className="resalt">
            {resalt}
          </div>
        )}
        {/* <div className="image">
          <img src="D:\תכנות יד\project2\react\picture" alt="w"/>
        </div> */}
      </div>
    </div>
  );
}

export default UploadFile;
