
/** Helper function to convert the cookie format */
const randomForestCookie = (c) => {
  return {
    name: c.name,
    value: c.value,
    domain: c.domain,
    expiry: c.expirationDate,
    secure: c.secure,
    httpOnly: c.httpOnly,
    javaScript: c.hostOnly
  }
}

/** Get all the cookies of the active tab */
const getCurrentTabCookies = async () => {
  let tab = await chrome.tabs.query({active: true, lastFocusedWindow: true});
  let cookies = await chrome.cookies.getAll({"url": tab[0].url});
  return cookies;
}

/** Detect authentication cookies using the random forest model */
const detectCookies = () => {
  let allCookies = [];
  getCurrentTabCookies().then(
    (cookies) => {
      allCookies = cookies.map(randomForestCookie);
      // alert(JSON.stringify(allCookies));

      // Send cookies to the Flask API
      fetch("http://127.0.0.1:5000/RF", {
        method: "POST",
        headers: {
          "Content-type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(allCookies)
      
      }).then(response => {
        if (response.ok) {
          return response.json()
        }
        else { alert("Something went wrong.") }
      
      }).then(authCookieNames => {
        // This is returning undefined, will fix later
        return authCookieNames
      })
    }
  );
}

/** 
 * Protect authentication cookies by setting Secure and 
 * HTTP-Only flags to true 
 * */ 
export const protectCookies = async () => {

  detectCookies().then(authCookieNames => {
    alert(authCookieNames);
    // Protection logic goes here
  });

}