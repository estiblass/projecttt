import { User } from '../types/User'
const userInitial: User ={id: '',
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
        lng: ''
    }
}
}
const userReducer = (state: User = userInitial, action: any) => {
    switch (action.type) {
        case 'SET_USER':
            state = action.data;
            state = { ...state }
            break;
        default:
            break;
    }
    return state;
}

export default userReducer

