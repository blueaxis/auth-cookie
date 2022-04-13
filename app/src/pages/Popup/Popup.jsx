import React from 'react';

import logo from '../../assets/img/icon-128.png';
import './Popup.css';
import { protectCookies } from './utils';

const Popup = () => {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>Protect authentication cookies by pressing the button.</p>
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
