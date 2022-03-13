import React from 'react';

import logo from '../../assets/img/icon-128.png';
import './Popup.css';
import { protectCookies } from './utils'

const Popup = () => {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Welcome! Edit this section at app/src/pages/Popup/Popup.jsx.
        </p>
      </header>
      <button
        onClick={protectCookies}
        className='Protect-button'
      >
        Secure Cookies
      </button>
    </div>
  );
};

export default Popup;
