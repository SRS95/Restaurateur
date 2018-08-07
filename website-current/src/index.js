import React from 'react';
import ReactDOM from 'react-dom';
import MyReservationsView from "./components/MyReservationsView/MyReservationsView.js";
import CustomerProfileView from "./components/CustomerProfileView/CustomerProfileView.js"
import './index.css';
import { BrowserRouter as Router, Route} from "react-router-dom";

class App extends React.Component {
    render() {
        return(
            <Router>
                <div>
                    <Route exact path="/" component={MyReservationsView}/>
                    <Route path="/customers/Marie Walters" component={CustomerProfileView}/>
                </div>
            </Router>
        );
    }
}

ReactDOM.render(<App />, document.getElementById('root'));
