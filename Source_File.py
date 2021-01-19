# -*- coding: utf-8 -*-
"""
Author: Yunyi Zhang
file: Source_File.py
times: 1/19/20219:16 AM
"""
from warcio.archiveiterator import ArchiveIterator
import re
import requests
import sys
import nltk

# nltk.download('wordnet') if there is no wordnet
from nltk.corpus import wordnet as wn
if len(sys.argv) > 1:
    file_name = sys.argv[1]


def loadData(file_name):
    stream = None
    if file_name.startswith("http://") or file_name.startswith(
            "https://"
    ):
        stream = requests.get(file_name, stream=True).raw
    else:
        stream = open(file_name, "rb")

    return stream


def findData(stream, regex, rawUrls = None, lenLimit = None):
    # for List object, must contain rawUrls
    entries = 0
    matching_entries = 0
    pages = []
    urls = []
    astream = stream if isinstance(stream, list) else ArchiveIterator(stream)
    for record in astream:
        entries = entries + 1
        if isinstance(stream, list):
            contents = record
        else:
            if record.rec_type == "warcinfo":
                continue
            if not ".com/" in record.rec_headers.get_header(
                    "WARC-Target-URI"
            ):
                continue
            contents = (
                record.content_stream().read().decode("utf-8", "replace")
            ).lower()
        m = regex.search(contents)
        if m:
            m = regex.search(contents, m.end())
            if m:
                matching_entries = matching_entries + 1
                pages.append(contents)
                curUrl = rawUrls[entries - 1] if isinstance(stream, list) else record.rec_headers.get_header("WARC-Target-URI")
                urls.append(curUrl)
        if lenLimit is not None and entries > lenLimit:
            break
    print(
        str(matching_entries)+ "/"+ str(entries)
    )
    return pages, urls, matching_entries

def writeFile(list, file_name):
    with open(file_name, 'w', encoding = 'utf-8') as filehandle:
        for item in list:
            filehandle.write('%s\n' % item)

def createURL(header, file_name):
    urlList = []
    with open(file_name, 'r', encoding = 'utf-8') as filehandle:
        lines = filehandle.readlines()
        for line in lines:
            urlList.append((header + line)[:-1])
    return urlList


if __name__ == '__main__':
    header = 'https://commoncrawl.s3.amazonaws.com/'
    readFileName = 's105r69Jbkhz'
    urlsName = 'urlsR.txt'
    # First we create search labels related to covid
    reCovid = re.compile(
        "covid|coronacoronavirus"
    )
    # Then we create search labels related to economy, the first several are summarized from some economic reports
    econWords = ['economic', 'crisis', 'income', 'wealth', 'business', 'employment', 'lower-wage', 'commerce', 'finance', 'buiness', 'payroll', 'sales']
    tendencyWords = ['increases', 'decreases', 'decline', 'damage', 'downturn', 'severe', 'contraction']
    strEcon = ""
    strTend = ""
    for word in econWords:
        strEcon += (word + '|')
    for word in tendencyWords:
        strTend += (word + '|')
    strEcon = strEcon[:-1]
    strTend = strTend[:-1]
    reEcon = re.compile(strEcon)
    reTend = re.compile(strTend)
    # Finally, perform two-stage searches finding the satisfactory webpages
    file_names = createURL(header, readFileName)
    count = 0
    lim = 1500
    allUrl = []
    for file_name in file_names:
        stream = loadData(file_name)
        # First stage
        pages, urls, add = findData(stream, reCovid)
    # Second stage
        pages, urls, add = findData(pages, reEcon, urls)
        pages, urls, add = findData(pages, reTend, urls)
        allUrl.extend(urls)
        count += add
        if count > lim:
            break
        writeFile(allUrl, urlsName)
    writeFile(allUrl, urlsName)




