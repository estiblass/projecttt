
import axios from "axios";
class EntryService {
  
  async getListUsers(){
    return axios.get('https://jsonplaceholder.typicode.com/users')
  }
  async getPostToUser(userId:string | null)
  {
    return axios.get(`https://jsonplaceholder.typicode.com/posts?userId=${userId}`)
  }
}

export default new EntryService()