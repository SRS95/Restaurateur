import React from "react";
import ReactDOM from "react-dom";
import "./Header.css"

export default class Header extends React.Component {
    // Note that correct-clear fixes the problem of the parent div not having a
    // height since all of its children are floating.
    render() {
        return(
            <div id="header">
                <div id="clear-fix">
                    <h1 id="kitchen-connect">KitchenConnect</h1>
                    <img id="logo" src="favicon.ico" alt="logo" />
                    <img id="stars" src="5_stars.png" alt="stars" />
                    <img id="analytics" src="analytics.png" alt="analytics" />
                    <img id="customer-outline" src="customer_outline.png" alt="customer outline" />
                    <div id="correct-clear" />
                </div>
            </div>
        );
    }
}

