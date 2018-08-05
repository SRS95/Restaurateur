import React from "react";
import ReactDOM from "react-dom";

import Header from "../Header/Header.js"
import "../Header/Header.css";

class ReservationRow extends React.Component {
    render() {
        return(
            <div id="row-rect-1" className="row-rect">
                <img className="prof-pic" src={this.props.photo} alt="Marie Walters Photo" />
                <h1 className="reservation-row-text">{this.props.name}</h1>
                <h1 className="reservation-row-text">{this.props.partySize} People</h1>
                <h1 className="reservation-row-text">{this.props.category}</h1>
                <img className="envelope" src="envelope.png" alt="Reservation Details" />
            </div>
        );
    }
}

class TableFilter extends React.Component {
    render() {
        return(
            <button id="reservations-dropdown-btn" className="dropdown-btn">
                <h1 id="reservations-dropdown-btn-text" className="dropdown-btn-text">
                    6:00pm - 7:00pm
                </h1>
            </button>
        );
    }
}

class FilterableReservationsTable extends React.Component {
    render() {
        return(
            <div>
                <TableFilter />
                <ReservationRow 
        name="Marie Walters" photo="marie.png" partySize="5" category="VIP"
                />
                <ReservationRow 
        name="Fannie Douglas" photo="fannie.png" partySize="3" category="First Time"
                />
                <ReservationRow 
        name="Eugina Hoffman" photo="eugina.png" partySize="6" category="Trending"
                />
            </div>
        );
    }
}

class FillTables extends React.Component {
    render() {
        return(
            <div>
                <h2 id="looking-light">Looking Light?</h2>
                <button id="fill-tables-btn" className="btn">
                    <h1 id="fill-tables-btn-text" className="btn-text">We can fill your tables!</h1>
                </button>
            </div>
        );
    }
}

export default class ReservationsView extends React.Component {
    render() {
        return(
            <body>
                <div>
                    <Header />
                </div>
                <body id="page-body">
                    <FillTables />
                    <FilterableReservationsTable />
                </body>
            </body>
        );
    }
}