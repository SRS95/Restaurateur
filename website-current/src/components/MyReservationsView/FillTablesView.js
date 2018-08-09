import React from "react";
import ReactDOM from "react-dom";
import Header from "../../components/Header/Header.js";
import "./FillTablesView.css";

class SelectionDisplay extends React.Component {
    render() {
        return(
            <div id="selection-display-box">
                <h1 className="selection-box-text"> Current Selection </h1>
                <table id="selection-table">
                    <tr>
                        <th></th>
                        <th>5:00pm</th>
                        <th>6:00pm</th>
                        <th>7:00pm</th>
                        <th>8:00pm</th>
                        <th>9:00pm</th>
                    </tr>
                    <tr>
                        <th>Very Small</th>
                        <td></td>
                        <td>5</td>
                        <td></td>
                        <td>5</td>
                        <td></td>
                    </tr>
                    <tr>
                        <th>Small</th>
                        <td>3</td>
                        <td></td>
                        <td></td>
                        <td>5</td>
                        <td></td>
                    </tr>
                    <tr>
                        <th>Medium</th>
                        <td></td>
                        <td>4</td>
                        <td>2</td>
                        <td></td>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Large</th>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td>2</td>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Big Group</th>
                        <td>2</td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td>1</td>
                    </tr>
                </table>
            </div>
        );
    }
}

class SelectionField extends React.Component {
    render() {
        return(
            <div className="field-outline">
                <form className="radio-btn-form">
                    <input type="radio" value={this.props.first} checked />{this.props.first}<br />
                    <input type="radio" value={this.props.second} />{this.props.second}<br />
                    <input type="radio" value={this.props.third} />{this.props.third}<br />
                    <input type="radio" value={this.props.fourth} />{this.props.fourth}<br />
                    <input type="radio" value={this.props.fifth} />{this.props.fifth}
                </form> 
            </div>
        );
    }
}

class NumberOfPartiesSelector extends React.Component {
    render() {
        return(
            <div className="selection-box"> 
                <h1 className="selection-box-text"> Number of Parties </h1>
                <div className="field-outline">
                    <form id="number-of-parties-form">
                        <input type="text" placeholder="Number of parties..." /><br />
                        <input type="submit" />
                    </form>
                </div>
            </div>
        );
    }
}

class PartySizeSelector extends React.Component {
    render() {
        return(
            <div className="selection-box"> 
                <h1 className="selection-box-text"> Select Party Size </h1>
                <SelectionField  
            first="Very Small (1-2)" second="Small (3-4)" third="Medium (5-6)" fourth="Large (7-9)" fifth="Big Group (10+)" 
                />
            </div>
        );
    }
}

class TimeSelector extends React.Component {
    render() {
        return(
            <div id="time" className="selection-box"> 
                <h1 className="selection-box-text"> Select Time </h1>
                <SelectionField 
            first="5:00pm" second="6:00pm" third="7:00pm" fourth="8:00pm" fifth="9:00pm" 
                />
            </div>
        );
    }
}

class TableSelector extends React.Component {
    render() {
        return(
            <div id="selector">
                <TimeSelector />
                <PartySizeSelector />
                <NumberOfPartiesSelector />
                <div className="correct-clear" />
            </div>
        );
    }
}

class FillTables extends React.Component {
    render() {
        return(
            <div id="table">
                <TableSelector />
                <SelectionDisplay />
                <button id="get-tables">
                    <h2 className="btn-text"> Get Tables! </h2>
                </button>
            </div>
        );
    }
}

export default class FillTablesView extends React.Component {
    render() {
        return(
            <div>
                <Header current="calendar" />
                <FillTables />
            </div>
        );
    }
}