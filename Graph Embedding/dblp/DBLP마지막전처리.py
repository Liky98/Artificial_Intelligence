"""
본 코드에서는 결측데이터가 있는 딕셔너리를 제거하여 최종 json파일을 완성함.

_id                 str
title               str
year                int
keywords            list of str
fos                 list of str
references          list of str
n_citation          int
abstract            str
authors._id         list of str
authors.name        list of str
authors.org         list of str
authors.orgid       list of str
venue.sid           str
venue.raw           str

14개의 Key
122783개의 Dictonory
374 MB

"""

import json

storeList = ["_id", "title", "year", "keywords", "fos", "references", "n_citation", "abstract", "authors._id",
     "authors.name", "authors.org", "authors.orgid", "venue.sid", "venue.raw"]

#n_citation은 len길이가 2이하라 뺴고 전처리
delList = ["_id", "title", "year", "keywords", "fos", "references", "abstract", "authors._id",
     "authors.name", "authors.org", "authors.orgid", "venue.sid", "venue.raw"]

with open("막전.json", 'r', encoding='utf-8') as f:
    data = f.read()
load_data = json.loads(data)
temp_data1 = load_data

index_list = list()
count = 0
index = 0

for i in temp_data1 :
    for j in delList :
        if len(str(i[j])) == 0 or len(str(i[j])) == 1 or len(str(i[j])) == 2 :
            index_list.append(index)
            break
    index +=1

for i in list(reversed(index_list)) :
    del temp_data1[i]

with open("DBLP최종.json", 'w') as f:
    json.dump(temp_data1, f, indent=4)

