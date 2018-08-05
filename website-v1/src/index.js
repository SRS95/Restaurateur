import React from "react";
import ReactDOM from "react-dom";

import ReservationsView from "./components/ReservationsView/ReservationsView.js"
import "./components/ReservationsView/ReservationsView.css";

import { Router, browserHistory } from "react-router";
import routes from "./routes";

class App extends React.Component {
    render() {
        return(
            <ReservationsView />
        );
    }
}

// ========================================

ReactDOM.render(<App />, document.getElementById("root"));