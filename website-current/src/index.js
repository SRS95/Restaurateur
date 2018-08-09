import React from 'react';
import ReactDOM from 'react-dom';
import Login from "./components/Login/Login.js";
import MyReservationsView from "./components/MyReservationsView/MyReservationsView.js";
import CustomerProfileView from "./components/CustomerProfileView/CustomerProfileView.js";
import FillTablesView from "./components/MyReservationsView/FillTablesView.js";
import AnalyticsView from "./components/AnalyticsView/AnalyticsView.js";
import './index.css';
import { BrowserRouter as Router, Route} from "react-router-dom";

class App extends React.Component {
    render() {
        return(
            <Router>
                <div>
                    <Route exact path='/' component={Login}/>
                    <Route exact path='/my-reservations' component={MyReservationsView}/>
                    <Route exact path="/my-reservations/fill-tables" component={FillTablesView} />
                    <Route exact path="/customers" component={CustomerProfileView}/>
                    <Route exact path="/customers/Marie Walters" component={CustomerProfileView}/>
                    <Route exact path="/analytics" component={AnalyticsView}/>
                </div>
            </Router>
        );
    }
}

ReactDOM.render(<App />, document.getElementById('root'));
