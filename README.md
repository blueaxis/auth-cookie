# Enhancing Web Authentication Security Using Random Forest

## Abstract

Building stateful web applications require a session mechanism to maintain server-side session state. Websites use HTTP cookies as authentication tokens to retain user credentials and keep them logged in between sessions. Its importance on authorization purposes makes it the primary target of attack as it allows intruders to gain access to features of an authenticated session. Previous attempts have been made to apply client-side protection mechanisms using authentication cookie detectors. However, such solutions rely on hand-coded rules based on empirical observations resulting in naive detectors. In this study, we aim to improve web security by selectively applying cookie attributes to authentication cookies detected using random forest methodology and assess its performance using machine learning model evaluation metrics. It is hypothesized that using random forest, the performance of previous detection algorithms will be enhanced, resulting to a secure web browsing environment.

## Requirements
\>= Python 3.9.10

	pip install -r requirements.txt

## Running

### Preparing the training data

	python model\features.py

### Training the model

	python model\train.py

### Running a Local API Instance

	python model\detect.py

## Attributions

Authentication Cookies Dataset

A. Casini, S. Calzavara, G. Tolomei

https://www.dais.unive.it/~calzavara/cookies-database.tar.gz