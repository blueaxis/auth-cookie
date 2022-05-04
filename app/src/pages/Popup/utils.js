/** Checks whether an object is iterable*/
const isIterable = (obj) => {
  if (obj == null) {
    return false;
  }
  return typeof obj[Symbol.iterator] === 'function';
};

/** Get all the cookies of the active tab */
const getCurrentTabCookies = async () => {
  let tab = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  let cookies = await chrome.cookies.getAll({ url: tab[0].url });
  // alert("Cookies in the site:\n" + JSON.stringify(cookies))
  return cookies;
};

/** Detect authentication cookies using the random forest model */
const detectCookies = async () => {
  return await getCurrentTabCookies().then(async (allCookies) => {
    // Send cookies to the Flask API
    return await fetch('http://127.0.0.1:5000/RF', {
      method: 'POST',
      headers: {
        'Content-type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(allCookies),
    })
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          alert('Something went wrong.');
        }
      })
      .then((authCookieNames) => {
        return authCookieNames;
      });
  });
};

/**
 * Protect authentication cookies by setting Secure and
 * HTTP-Only flags to true
 * */
export const protectCookies = () => {
  detectCookies().then((authCookieNames) => {
    if (!isIterable(authCookieNames)) {
      alert('No authentication cookies were detected.');
      return;
    }

    for (let c of authCookieNames) {
      // This will set the Secure and HTTP-Only flags to true
      let authCookie = {
        name: c.name,
        value: c.value,
        domain: c.domain,
        path: c.path,
        expirationDate: c.expirationDate,
        secure: true,
        httpOnly: true,
        sameSite: c.sameSite,
        storeId: c.storeId,
        url: 'https://' + c.domain.slice(1, c.domain.length),
      };
      chrome.cookies.set(authCookie);
    }
    alert('Authentication cookies are now protected.');
  });
};

export const deleteCookies = () => {
  fetch('http://127.0.0.1:5000/delete', {
    method: 'POST',
    headers: {
      'Content-type': 'application/json',
      Accept: 'application/json',
    },
  })
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        alert('Something went wrong.');
      }
    })
    .then((authCookieNames) => {
      for (let c of authCookieNames) {
        let authCookie = {
          name: c.name,
          storeId: c.storeId,
          url: "https://" + c.domain.slice(1, c.domain.length),
        };
        chrome.cookies.remove(authCookie);
      }
      alert('Authentication cookies deleted.');
    });
};
