# Convert dataset values into training features
# TODO:
# 1. Rewrite this as a function that can be called in main
# 2. Move data normalization in this file


import pandas as pd
import numpy as np
from collections import Counter


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

STANDARD_NAMES = ["JSESSIONID", "ASPSESSIONIDXXXXXXXX", "ASP.NET_SessionId", "PHPSESSION", "wiki18_session",
                    "WebLogicSession", "BIGipServerxxx_xxx_xxx_PROTO", "SERVERID", "SaneID", "ssuid", "vgnvisitor",
                    "SESSION_ID", "NSES40Session", "iPlanetUserId", "gx_session_id_", "JROUTE", "RMID", "JSESSIONID",
                    "Apache", "CFID", "CFTOKEN", "CFGLOBALS", "RoxenUserID", "JServSessionIdroot", "sesessionid",
                    "PD-S-SESSION-ID", "PD_STATEFUL", "WEBTRENDS_ID", "__utmX", "sc_id", "s_sq", "s_sess", "s_vi_XXXXXX",
                    "MintUnique", "MintXXXX", "SS_X_CSINTERSESSIONID", "CSINTERSESSIONID", "_sn", "BCSI-CSCXXXXXXX",
                    "Ltpatoken", "Ltpatoken2", "LtpatokenExpiry", "LtpatokenUsername", "DomAuthSessID", "connect.sid"]

if __name__ == "__main__":

# *******************************************
# Feature conversion
# *******************************************

# Read dataset from CSV file
    dataset = pd.read_csv("data/cookies.csv", names=[
        "Website", "ID", "Name", "Value", "Domain", "Path", "Secure", "Expiry", "HTTP_Only", "JavaScript", "Class"])

# Naming Scheme
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

def cookieCutter(x):
    dataset = pd.DataFrame(x)
    ret = pd.DataFrame()

    ret["Expiry"] = dataset["expirationDate"].map(lambda e: e - 1370000000).fillna(0)
    ret["JavaScript"] = dataset.hostOnly.map(lambda j: 1 if (j == "True") else 0)
    ret["Scheme"] = dataset["name"].map(lambda n: 1 if (n in STANDARD_NAMES) else 0)
    ret["IC"] = np.asarray(dataset["value"].map(index_of_coincidence)).astype(np.float32)
    ret["Entropy"] = np.asarray(dataset["value"].map(shannon_entropy)).astype(np.float32)
    ret["Length"] = dataset["value"].map(lambda v: len(str(v)))

    length_mean = ret.Length.mean()
    length_std = ret.Length.std()

    ret["Z_Length"] = ret.apply(lambda z:
                                    (z.Length - length_mean) / length_std
                                    ,axis=1).fillna(0)
    group_by_secure = dataset.groupby(["domain", "secure"]).size()
    group_by_http = dataset.groupby(["domain", "httpOnly"]).size()
    group_by_js = dataset.groupby(["domain", "hostOnly"]).size()
    ret["TFIDF_S"] = dataset.apply(lambda w: w.secure * idf_(
    w.domain, group_by_secure), axis=1)
    ret["TFIDF_H"] = dataset.apply(lambda w: w.httpOnly * idf_(
    w.domain, group_by_http), axis=1)
    ret["TFIDF_J"] = dataset.apply(lambda w: w.hostOnly * idf_(
    w.domain, group_by_js, js=True), axis=1)

    #normalize
    ret["Expiry"] = (ret["Expiry"] - ret["Expiry"].mean()) / ret["Expiry"].std()
    ret["IC"] = (ret["IC"] - ret["IC"].mean()) / ret["IC"].std()
    ret["Entropy"] = (ret["Entropy"] - ret["Entropy"].mean()) / ret["Entropy"].std()
    return ret