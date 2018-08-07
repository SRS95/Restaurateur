import React from "react";
import ReactDOM from "react-dom";
import Header from "../Header/Header.js"
import "./MyReservationsView.css"
import CustomerProfileView from "../CustomerProfileView/CustomerProfileView.js"
import { BrowserRouter as Router, Link, Route } from "react-router-dom";

class Row extends React.Component {
    render() {
        return(
            <div className="row-container">
                <Link to={`/customers/${this.props.name}`}>
                    <img className="prof-pic" src={`/${this.props.photo}`} alt="Marie Walters Photo" />
                    <h1 className="customer-name">{this.props.name}</h1>
                </Link>
                <h1 className="reservation-row-text">{this.props.partySize} People</h1>
                <h1 className="reservation-row-text">{this.props.category}</h1>
                <img className="envelope" src={`/envelope.png`} alt="Reservation Details" />
                <div className="correct-clear" />
            </div>
        );
    }
}

class Table extends React.Component {
    render() {
        return(
            <div id="table-container">
                <Row photo="marie.png" name="Marie Walters" partySize="5" category="VIP"/>
                <Row name="Luigi Balbo Bertone di Sambuy" photo="fannie.png" partySize="3" category="First Time" />
                <Row name="Eugina Hoffman" photo="eugina.png" partySize="6" category="Trending" />
            </div>
        );
    }
}

class TableFilter extends React.Component {
    render() {
        return(
            <div id="table-filter-container">
                <button id="reservations-dropdown-btn" className="btn">
                    <h1 id="reservations-dropdown-btn-text" className="btn-text">
                        6:00pm - 7:00pm
                    </h1>
                </button>
            </div>
        );
    }
}

class FillTables extends React.Component {
    render() {
        return(
            <div id="fill-tables">
                <h2 id="looking-light">Looking Light?</h2>
                <button id="fill-tables-btn" className="btn">
                    <h1 id="fill-tables-btn-text" className="btn-text">We can fill your tables!</h1>
                </button>
            </div>
        );
    }
}

class Reservations extends React.Component {
    render() {
        return(
            <div id="reservations">
                <FillTables />
                <TableFilter />
                <Table />
            </div>
        );
    }
}

export default class MyReservationsView extends React.Component {
    render() {
        return(
            <div>
                <Header />
                <Reservations />
            </div>
        );
    }
}