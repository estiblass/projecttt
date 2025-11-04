import { createStore } from "redux";
import { devToolsEnhancer } from "redux-devtools-extension";
import rootReducer from "./rootReducer";
import { User } from "../types/User";
const store = createStore(rootReducer, devToolsEnhancer({}));

export type StoreType = ReturnType<typeof rootReducer>
export default store