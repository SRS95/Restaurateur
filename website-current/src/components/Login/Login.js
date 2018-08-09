import React from 'react';
import ReactDOM from 'react-dom';
import { Link } from "react-router-dom";
import "./Login.css"

export default class Login extends React.Component {
    render() {
        return(
            <div class="container">
                <label for="uname"><b>Username</b></label>
                <input type="text" placeholder="Enter Username" name="uname" required />
        
                <label for="psw"><b>Password</b></label>
                <input type="password" placeholder="Enter Password" name="psw" required />
                
                <button type="submit">Login</button>
            </div>
        );
    }
}