import json

with open("DBLP최종.json", 'r', encoding='utf-8') as f:
    data = f.read()
load_data = json.loads(data)

storeList = ["_id", "title", "year", "keywords", "fos", "references", "n_citation", "abstract", "authors._id",
     "authors.name", "authors.org", "authors.orgid", "venue.sid", "venue.raw"]

for i in storeList :
    print("{0}의 데이터 타입 = {1}".format(i,type(load_data[0][i])))

print("데이터의 개수는 {} 입니다.".format(len(load_data)))

"""
_id의 데이터 타입 = <class 'str'>
title의 데이터 타입 = <class 'str'>
year의 데이터 타입 = <class 'int'>
keywords의 데이터 타입 = <class 'list'>
fos의 데이터 타입 = <class 'list'>
references의 데이터 타입 = <class 'list'>
n_citation의 데이터 타입 = <class 'int'>
abstract의 데이터 타입 = <class 'str'>
authors._id의 데이터 타입 = <class 'list'>
authors.name의 데이터 타입 = <class 'list'>
authors.org의 데이터 타입 = <class 'list'>
authors.orgid의 데이터 타입 = <class 'list'>
venue.sid의 데이터 타입 = <class 'str'>
venue.raw의 데이터 타입 = <class 'str'>
데이터의 개수는 122783 입니다.
"""