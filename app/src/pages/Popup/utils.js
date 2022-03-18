
const getCurrentTabCookies = async () => {
  let tab = await chrome.tabs.query({active: true, lastFocusedWindow: true});
  let cookies = await chrome.cookies.getAll({"url": tab[0].url});
  return cookies;
}

// Detect authentication cookies using the random forest model
const detectCookies = () => {
  let authCookieNames = [];
  getCurrentTabCookies().then(
    (cookies) => {
      alert("Number of cookies in this site: " + cookies.length)
      // Send cookies to the Flask API
    }
  );

  // Receive cookies from the API
  return authCookieNames;
}

// Protect authentication cookies by setting Secure and
// HTTP-Only flags to true
export const protectCookies = async () => {

  let authCookieNames = detectCookies();
  // protection logic goes here
}