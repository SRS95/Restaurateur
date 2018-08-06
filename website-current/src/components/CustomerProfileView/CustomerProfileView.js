import React from "react";
import ReactDOM from "react-dom";
import Header from "../Header/Header.js"
import "./CustomerProfileView.css"

class Instructions extends React.Component {
    render() {
        return(
            <div id="instructions">
                <h1>instructions</h1>
            </div>
        );
    }
}

class Palate extends React.Component {
    render() {
        return(
            <div id="palate">
                <h1 id="palate-title">Marie's Palate:</h1>
            </div>
        );
    }
}

class Bio extends React.Component {
    render() {
        return(
            <div id="bio">
                <h1 id="bio-title"> Marie's Bio</h1>
                <p id="bio-text">
                    Just a girl on the lookout for the
                    world's best Panna Cotta.
                </p>
            </div>
        );
    }
}

class KeyInfo extends React.Component {
    render() {
        return(
            <div id="key-info">
                <img id="prof-pic" src="marie.png" alt="Marie's photo" />
                <h1 id="customer-name">Marie Walters</h1>
                <p className="customer-subtext">mariewalters@gmail.com</p>
                <p className="customer-subtext">Last visit: May 7</p>
            </div>
        );
    }
}

class Profile extends React.Component {
    render() {
        return(
            <div id="profile">
                <KeyInfo />
                <Bio />
                <Palate />
                <Instructions />
            </div>
        );
    }
}

export default class CustomerProfileView extends React.Component {
    render() {
        return(
            <div>
                <Header />
                <Profile />
            </div>
        );
    }
}