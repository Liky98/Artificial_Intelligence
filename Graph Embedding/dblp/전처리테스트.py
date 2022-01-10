import json
import pandas as pd
import gc
import pickle
import re
from collections import OrderedDict
import pprint
import time
count = 0
DBLP = list()
tempList = OrderedDict()


find_list = ["_id", "title", "authors", "venue", "year", "keywords",
             "fos", "references", "n_citation", "publisher", "abstract"]

first_list = ["_id", "title", "year", "keywords",
             "fos", "references", "n_citation", "publisher", "abstract"]

del_list = ["page_start","page_end", "doc_type", "lang", "volume",
            "issue", "issn", "isbn", "doi", "pdf", "url", "indexed_abstract"]


authors_sublist = ["_id", "name", "org", "orgid"]
venue_sublist = ["_id", "raw"]

def add_data() :
    global DBLP, load_data, count, find_list, first_list, del_list, authors_sublist, venue_sublist
    TF = True
    tempList = OrderedDict()


    # 필요 Keys 값 전부 있는지 확인
    for i in find_list :
        if i not in load_data[count] :
            TF= False
            break

    # authors, venue 제외하고 나머지 결측값 없는 데이터만 뽑기
    for x in first_list:
        if x not in load_data[count] :
            TF = False
            break
        tempList[x] = load_data[count][x]

    # authors id, name, org, orgid 만 뽑고 결측값 없는 데이터만 처리
    for xx in authors_sublist:
        if not xx in load_data[count]["authors"][0]:
            TF = False
            break
    for i in load_data[count]["authors"] :
        tempList["authors"][xx] = i[xx]

    # venue id, raw 만 뽑고 결측값 없는 데이터만 처리
    for xxx in venue_sublist:
        if not xxx in load_data[count]["venue"] :
            TF = False
            break
    for i in load_data[count]["venue"]:
        tempList["venue"][xxx] = i[xxx]



    # 완벽한 애들만 리스트에 추가 저장
    if TF :
        DBLP.append(tempList)

with open("2차 전처리/2차 전처리 {0}번째.json".format(1), 'r', encoding='utf-8') as f:
    data = f.read()
fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data)  # NumberInt(0) 변환하기
load_data = json.loads(fixed_data)
print("{0}번째 Json파일을 열었습니다.".format(1))

try:
    while 1:
        add_data()
        count += 1
except IndexError:
    print("작업완료")

with open("테스트{0}번째.json".format(1), 'w') as f:
    json.dump(DBLP, f, indent=4)

#%%
for xx in authors_sublist:
    print(xx)
    tempcount = 0
    try:
        while 1 :
            print(load_data[0]["authors"][tempcount][xx])
            tempcount +=1
    except IndexError :
        print("인덱스")
#    print(load_data[0]["authors"].keys())
#%%
for i in load_data[0]["authors"] :
    print(i['_id'])