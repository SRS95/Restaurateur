import React from "react";
import ReactDOM from "react-dom";
import "./Header.css"
import { Link } from "react-router-dom";

export default class Header extends React.Component {
    // Note that correct-clear fixes the problem of the parent div not having a
    // height since all of its children are floating.
    render() {
        return(
            <div id="header">
                <div id="clear-fix">
                    <Link to={'/my-reservations'}>
                        <h1 id="kitchen-connect">KitchenConnect</h1>
                        <img id="logo" src={`/favicon.ico`} alt="logo" />
                    </Link>
                        <img id="stars" src={this.determineStarsSrc()} alt="stars" />
                    <Link to={'/analytics'}>
                        <img id="analytics" src={this.determineAnalyticsrSrc()} alt="analytics" />
                    </Link>
                    <Link to={"/customers/Marie Walters"}>
                        <img id="customer-outline" src={this.determineCustomerOutlineSrc()} alt="customer outline" />
                    </Link>
                    <Link to={'/my-reservations'}>
                        <img id="calendar" src={this.determineCalendarSrc()} alt="reservations" />
                    </Link>
                    <div id="correct-clear" />
                </div>
            </div>
        );
    }

    determineCalendarSrc() {
        if (this.props.current == "calendar") return "/images/header/calendar_line.png"
        else return "/images/header/calendar.png"
    }

    determineCustomerOutlineSrc() {
        if (this.props.current == "customer") return "/images/header/customer_outline_line.png"
        else return "/images/header/customer_outline.png"
    }

    determineAnalyticsrSrc() {
        if (this.props.current == "analytics") return "/images/header/analytics_line.png"
        else return "/images/header/analytics.png"
    }

    determineStarsSrc() {
        if (this.props.current == "stars") return "/images/header/5_stars_line.png"
        else return "/images/header/5_stars.png"
    }
}