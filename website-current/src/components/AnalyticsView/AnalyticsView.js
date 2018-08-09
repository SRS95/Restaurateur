import React from 'react';
import ReactDOM from 'react-dom';
import { Link } from "react-router-dom";
import Header from "../Header/Header.js"
import './AnalyticsView.css';

class AnalyticsDisplay extends React.Component {
    render() {
        if(this.props.currentChoice == "Engagement") {
            return this.displayEngagement();
        } else if (this.props.currentChoice == "Sentiment") {
            return this.displaySentiment();
        } else if (this.props.currentChoice == "Patrons") {
            return this.displayPatrons();
        } else {
            return this.displayFood();
        }
    }

    displayEngagement() {
        return(
            <div className="graphs-container">
                <img className="graph-header" 
            src="/images/analytics/engagement/money-header.png"
            alt="engagement graph header" /><br />
                <img className="graph" 
            src="/images/analytics/engagement/money.png"
            alt="engagement graph" /><br />
                <img className="graph-header" 
            src="/images/analytics/engagement/visits-header.png"
            alt="engagement graph header" /><br />
                <img className="graph" 
            src="/images/analytics/engagement/visits.png"
            alt="engagement graph" /><br />
                <img className="graph-header" 
            src="/images/analytics/engagement/sentiment-header.png"
            alt="engagement graph header" /><br />
                <img className="graph" 
            src="/images/analytics/engagement/sentiment.png"
            alt="engagement graph" /><br />
            </div>
        );
    }

    displaySentiment() {
        return(
            <div className="graphs-container">
                <img className="graph-header" 
            src="/images/analytics/sentiment/header.png"
            alt="sentiment graph header" /><br />
                <img className="graph" 
            src="/images/analytics/sentiment/graph.png"
            alt="sentiment graph" /><br />
            </div>
        );
    }

    displayPatrons() {
        return(
            <div className="graphs-container">
                <img className="graph-header" 
            src="/images/analytics/patrons/customer-growth-header.png"
            alt="engagement graph header" /><br />
                <img className="graph" 
            src="/images/analytics/patrons/customer-growth.png"
            alt="engagement graph" /><br />
                <img className="graph" 
            src="/images/analytics/patrons/time-waiting.png"
            alt="engagement graph" /><br />
                <img className="graph" 
            src="/images/analytics/patrons/table-time.png"
            alt="engagement graph" /><br />
            </div>
        );
    }

    displayFood() {
        return(
            <div className="graphs-container">
                <img className="graph-header" 
            src="/images/analytics/food/header.png"
            alt="sentiment graph header" /><br />
                <img className="graph" 
            src="/images/analytics/food/graph.png"
            alt="sentiment graph" /><br />
            </div>
        );
    }

}

class AnalyticsControl extends React.Component {
    render() {
        return(
            <div id="control-container">
                <div id="control-header">
                    <h2 id="panel-title">KitchenAnalytics</h2>
                </div>
                <div id="control-body">
                    
                    <button onClick={() => this.props.onClick("Engagement")}
                style={{opacity: this.parseID(this.props.currentChoice, "Engagement")}}
                className="control-button">Engagement</button><br />
                    
                    <button onClick={() => this.props.onClick("Sentiment")}  
                style={{opacity: this.parseID(this.props.currentChoice, "Sentiment")}}
                className="control-button">Sentiment</button><br />
                    
                    <button onClick={() => this.props.onClick("Patrons")} 
                style={{opacity: this.parseID(this.props.currentChoice, "Patrons")}}
                className="control-button">Patrons</button><br />
                    
                    <button onClick={() => this.props.onClick("Food")} 
                style={{opacity: this.parseID(this.props.currentChoice, "Food")}} 
                className="control-button" id="last-button">Food</button>
                
                </div>
            </div>
        );
    }

    parseID(currentChoice, buttonName) {
        if(currentChoice == buttonName) { 
            return 0.5;
        } else {
            return 1.0;
        }
    }
}

class Analytics extends React.Component {
    constructor(props) {
        super(props);
        this.state = { currentChoice: "Engagement" };
    }

    render() {
        return(
            <div id="analytics-container">
                <AnalyticsControl 
                    currentChoice={this.state.currentChoice}
                    onClick={(choice) => this.handleClick(choice)} 
                />
                <AnalyticsDisplay
                    currentChoice={this.state.currentChoice}
                />
            </div>
        );
    }

    handleClick(choice) {
        this.setState({currentChoice: choice});
    }
}

export default class AnalyticsView extends React.Component {
    render() {
        return(
            <div>
                <Header current="analytics" />
                <Analytics />
            </div>
        );
    }
}