# Convert dataset values into training features
# TODO:
# 1. Rewrite this as a function that can be called in main
# 2. Move data normalization in this file


import pandas as pd
import numpy as np
from collections import Counter


# Read dataset from CSV file
dataset = pd.read_csv("data/cookies.csv", names=[
    "Website", "ID", "Name", "Value", "Domain", "Path", "Secure", "Expiry", "HTTP_Only", "JavaScript", "Class"])


# Helper functions for feature conversion
# *******************************************


def index_of_coincidence(c_value):
    c_value = str(c_value)
    ctr = Counter(c_value)
    n = len(c_value)
    numerator = 0
    for c in "abdefghijklmnqrstuvxyzABDEFGHIJKLMNQRSTUVXYZ0123456789!#$%&'*+-.^_`|~":
        numerator += ctr[c] * (ctr[c] - 1)
    try:
        return numerator / (n * (n - 1))
    except ZeroDivisionError:
        return 1


def shannon_entropy(c_value, base=2):
    c_value = str(c_value)
    ctr = Counter(c_value)
    n = len(c_value)
    entropy = 0
    for _, c in ctr.items():
        entropy += (c / n) * np.log(c / n) / np.log(base)
    return -entropy


def idf_(website, group, js=False):
    tf_true, tf_false = 1, 0
    if js:
        tf_true, tf_false = 0, 1
    n, nt = 0, 0
    try:
        n = group[(website, tf_false)] + group[(website, tf_true)]
        nt = group[(website, tf_true)]
    except KeyError:
        try:
            n = group[(website, tf_false)]
        except KeyError:
            n = group[(website, tf_true)]

    return np.log(n / (nt + 1)) / np.log(2)


# *******************************************


# Feature conversion
# *******************************************

# Naming Scheme
STANDARD_NAMES = ["JSESSIONID", "ASPSESSIONIDXXXXXXXX", "ASP.NET_SessionId", "PHPSESSION", "wiki18_session",
                  "WebLogicSession", "BIGipServerxxx_xxx_xxx_PROTO", "SERVERID", "SaneID", "ssuid", "vgnvisitor",
                  "SESSION_ID", "NSES40Session", "iPlanetUserId", "gx_session_id_", "JROUTE", "RMID", "JSESSIONID",
                  "Apache", "CFID", "CFTOKEN", "CFGLOBALS", "RoxenUserID", "JServSessionIdroot", "sesessionid",
                  "PD-S-SESSION-ID", "PD_STATEFUL", "WEBTRENDS_ID", "__utmX", "sc_id", "s_sq", "s_sess", "s_vi_XXXXXX",
                  "MintUnique", "MintXXXX", "SS_X_CSINTERSESSIONID", "CSINTERSESSIONID", "_sn", "BCSI-CSCXXXXXXX",
                  "Ltpatoken", "Ltpatoken2", "LtpatokenExpiry", "LtpatokenUsername", "DomAuthSessID", "connect.sid"]
dataset["Scheme"] = dataset["Name"].map(lambda n: 1 if (n in STANDARD_NAMES) else 0)

# Expiration
dataset["Expiry"] = dataset["Expiry"].map(lambda e: e - 1370000000)

# Index of Coincidence
dataset["IC"] = np.asarray(dataset["Value"].map(index_of_coincidence)).astype(np.float32)

# Shannon Entropy
dataset["Entropy"] = np.asarray(dataset["Value"].map(shannon_entropy)).astype(np.float32)

# Length
dataset["Length"] = dataset["Value"].map(lambda v: len(str(v)))

# Z-Length
length_mean = dataset.groupby(["Website"])[["Length"]].mean()
length_std = dataset.groupby(["Website"])[["Length"]].std()
dataset["Z_Length"] = dataset.apply(lambda z:
                                    (z.Length - length_mean["Length"].loc[
                                        z.Website]) / length_std["Length"].loc[z.Website]
                                    , axis=1).fillna(0)

# TF-IDFs
group_by_secure = dataset.groupby(["Website", "Secure"]).size()
group_by_http = dataset.groupby(["Website", "HTTP_Only"]).size()
group_by_js = dataset.groupby(["Website", "JavaScript"]).size()

dataset["TFIDF_S"] = dataset.apply(lambda w: w.Secure * idf_(
    w.Website, group_by_secure), axis=1)
dataset["TFIDF_H"] = dataset.apply(lambda w: w.HTTP_Only * idf_(
    w.Website, group_by_http), axis=1)
dataset["TFIDF_J"] = dataset.apply(lambda w: w.JavaScript * idf_(
    w.Website, group_by_js, js=True), axis=1)

# *******************************************


# Remove unused columns
dataset = dataset.drop(
    columns=["Website", "ID", "Name", "Value", "Domain", "Path", "Secure", "HTTP_Only"])

# Save as csv file
dataset.to_csv("./data/features.csv", encoding='utf-8', index=False)

#cookie feature converter
#cookie =  {"domain": DOMAIN,
    # "expirationDate": EXPIRY,
    # "hostOnly": IS_JAVASCRIPT,
    # "httpOnly": IS_HTTP_ONLY,
    # "name": NAME,
    # "path": PATH,
    # "sameSite": IS_SAMESITE,
    # "secure": IS_SECURE,
    # "storeId": ID,
    # "value":VALUE}
 # ret = {Expiry 	JavaScript 	Scheme 	IC 	Entropy 	Length 	Z_Length 	TFIDF_S 	TFIDF_H 	TFIDF_J}

def cookieCutter(cookie):
    ret = {}
    ret["Expiry"] = cookie["expirationDate"] - 1370000000
    ret["JavaScript"] = cookie["hostOnly"]
    ret["Scheme"] = (1 if (cookie["name"] in STANDARD_NAMES) else 0)
    ret["IC"] = index_of_coincidence(cookie["value"])
    ret["Entropy"] = shannon_entropy(cookie["value"])
    ret["Length"] = len(str(cookie['value']))
    ret["Z_Length"] = -2.8039 #average ng traingin dataset
    ret["TFIDF_S"] = 0.099413877 #average tfidf_s ng training dataset
    ret["TFIDF_H"] = 0.20152078 #average tfidf_h ng training dataset
    ret["TFIDF_J"] = 0.470163575 #average tfidf_j ng training dataset
    #normalize
    ret["Expiry"] = (ret["Expiry"] - 146540488.5) / 578423638.8
    ret["JavaScript"] = (ret["JavaScript"] - 0.418695994) / 0.493345375
    ret["Scheme"] = (ret["Scheme"] - 0.029850746) / 0.170175437
    ret["IC"] = (ret["IC"] - 0.18569332) / 0.319782136
    ret["Entropy"] = (ret["Entropy"] - 2.956862998) / 1.581445464

    return [
        ret["Expiry"],
        ret["JavaScript"],
        ret["Scheme"],
        ret["IC"],
        ret["Entropy"],
        ret["Length"],
        ret["Z_Length"],
        ret["TFIDF_S"],
        ret["TFIDF_H"],
        ret["TFIDF_J"]
    ]