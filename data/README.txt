1. Title: Authentication Cookies Dataset

2. Sources:
	(a) Creator: A. Casini, S. Calzavara, G. Tolomei
	(b) Date: October, 2014

3. Past Usage:

	S. Calzavara, G. Tolomei, A. Casini, M. Bugliesi and S.Orlando
	"A Supervised Learning Approach to Protect Web Authentication"
	Università Ca' Foscari, Venezia
	
	The data was used to extract relevant information about cookies 
	in order to devise a novel authentication cookie detector, 
	protecting the user against session hijacking attacks.

4. Relevant Information:
	--- This is the first known database of (non-)authentication cookies. An
	authentication cookie is a cookie that contains the authentication information
	needed to preserve the user web session.

5. Number of Instances: 2546

6. Number of attributes: 11

7. Attributes Information:
	1) website
	the name of the website

	2) id
	the cookie id

	3) name
	the cookie name
	
	4) value
	the cookie value

	5) domain
	the domain of the cookie

	6) path
	the path set for the cookie

	7) secure
	presence/absence of the Secure flag
	
	8) expiry
	the cookie expiration date (in Unix format)
	
	9) httponly
	presence/absence of the Http-Only flag
	
	10) js
	determines if the cookie has been set by JavaScript
	
	11) class
	the cookie class (0: non-authentication, 1: authentication)

8. Missing Attribute Values:
	None

9. Class Distribution: number of instances per class
	class 0 2204
	class 1 342

10. Extras: 'cookies.db'
	--- We also include in the 'db' directory the Sqlite database 'cookies.db', used in our
	script to store the cookies together with the authentication tokens they belong to.
