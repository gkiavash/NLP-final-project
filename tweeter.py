import os.path
import re

import pandas as pd
import tweepy as tweepy


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def replace_all(text: str, dic: dict):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


# replace chars
url_dict = {
    "\n": " ",
    "&amp;": "&",
    "&gt;": ">",
    "&lt;": "<"
}
# regex for removal of URLs in Tweets
regex = r"http\S+"

consumer_key = "vujt2Ip28lIcMFkHXkyEzuuEM"
consumer_secret = "JDIL6gO4YKO6RzZVc849IPbBEs1GqlwOpcc3rCtUQwRMXIBYJT"
access_token = "1510215684986392576-noV89oJq1CeP5w69aYxrbGBYkjiwQt"
access_token_secret = "LO8ceOG38OsFfpmtUmEk0GhjWZZVSrFt0Dtlzm6gye7d6"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

inputpath = "https://raw.githubusercontent.com/kglandt/stance-detection-in-covid-19-tweets/main/dataset/face_masks_train.csv"
outputpath = "./face_masks_train_retrieved.tsv"


def add_full_text(inputpath, outputpath):
    df = pd.read_csv(inputpath, dtype=str)
    #  read data
    id_list = list(df['Tweet Id'].values)
    full_texts = ["" for i in range(len(id_list))]
    error = 0

    for chunk in chunks(id_list, 100):
        try:
            results = api.lookup_statuses(chunk, tweet_mode='extended')
            print(len(results))
        except Exception as e:
            print(e)

        for res in results:
            # get index of respective tweet ID
            id_idx = id_list.index(res._json["id_str"])
            # replace web symbols and remove URLs
            txt = re.sub(regex, '', res._json['full_text']).strip()
            txt = replace_all(txt, url_dict)
            full_texts[id_idx] = txt

    df["full_text"] = full_texts
    df.to_csv(outputpath, sep="\t", index=False)  # replace dataset_full path with expertfile


from os import listdir
from os.path import isfile, join
mypath = 'dataset'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)

for path in onlyfiles:
    add_full_text(os.path.join('dataset', path), os.path.join('dataset_full', path))
