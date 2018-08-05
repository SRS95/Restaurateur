import React from "react";
import ReactDOM from "react-dom";

class NavIcons extends React.Component {
    render() {
        return(
            <div>
                <img id="stars" src="5_stars.png" alt="5 stars"></img>
                <img id="analytics" src="analytics.png" alt="analytics"></img>
                <img id="customer-outline" src="customer_outline.png" alt="customer outline"></img>
            </div>
        );
    }
}

export default class Header extends React.Component {
    render() {
        return(
            <div>
                <h1 id="kitchen-connect">KitchenConnect</h1>
                <img id="logo" src="logo.png" alt="logo" />
                <NavIcons />
            </div>
        );
    }
}