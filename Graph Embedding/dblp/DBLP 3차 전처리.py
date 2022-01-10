"""
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
153164개의 Dictonory

"""
import json
import re
from collections import OrderedDict

count = 0
DBLP = list()



find_list = ["_id", "title", "authors", "venue", "year", "keywords",
             "fos", "references", "n_citation", "abstract"]

#author 랑 venue 빼고
first_list = ["_id", "title", "year", "keywords",
             "fos", "references", "n_citation", "abstract"]

del_list = ["page_start","page_end", "doc_type", "lang", "volume",
            "issue", "issn", "isbn", "doi", "pdf", "url", "indexed_abstract"]

authors_sublist = ["_id", "name", "org", "orgid"]
venue_sublist = ["sid", "raw"]

check_list = ["_id", "title", "year", "n_citation", "abstract"]

def add_data() :
    global DBLP, load_data, count, find_list, first_list, del_list, authors_sublist, venue_sublist,check_list
    TF = True
    tempList = OrderedDict()
    authors_tempList = list()
    try:
        # 필요 Keys 값 전부 있는지 확인
        for i in find_list :
            if i in load_data[count] :
                TF = True
                #빈 데이터값 확인
                if len(str(load_data[count][i])) == 0 :
                    TF=False
                if load_data[count][i] == "[]" :
                    TF=False
            else :
                TF= False
                break


        # authors, venue 제외하고 나머지 결측값 없는 데이터만 뽑기
        if TF :
            for x in first_list:
                tempList[x] = load_data[count][x]


        # authors _id, name, org, orgid 만 뽑고 결측값 없는 데이터만 처리
        if TF :
            for xx in authors_sublist:
                if xx in load_data[count]["authors"][0]:
                    TF = True
                else:
                    TF = False
                    break

        if TF :
            for xx in authors_sublist :
                for i in load_data[count]["authors"] :
                    authors_tempList.append(i[xx])
                tempList["authors.{}".format(xx)] = authors_tempList
                authors_tempList = list()

        # venue _id, raw 만 뽑고 결측값 없는 데이터만 처리
        if TF :
            for xxx in venue_sublist:
                if xxx in load_data[count]["venue"]:
                    TF = True
                else:
                    TF = False
                    break
        if TF :
            for xxx in venue_sublist:
                tempList["venue.{}".format(xxx)] = load_data[count]["venue"][xxx]
    except KeyError :
        TF = False
    except IndexError :
        TF = False

    # 완벽한 애들만 리스트에 추가 저장
    if TF :
        DBLP.append(tempList)


for i in range(17):
    count = 0
    with open("1차 전처리/처리완료 {}번째.json".format(i+1), 'r', encoding='utf-8') as f:
        data = f.read()
    fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data)  # NumberInt(0) 변환하기
    load_data = json.loads(fixed_data)
    print("{}번째 Json파일을 열었습니다.".format(i+1))

    for _ in load_data :
        add_data()
        count +=1


with open("막전.json", 'w') as f:
    json.dump(DBLP, f, indent=4)

#%%
# 최종 확인 데이터
storeList = ["_id", "title", "year", "keywords", "fos", "references", "n_citation", "abstract", "authors._id",
     "authors.name", "authors.org", "authors.orgid", "venue.sid", "venue.raw"]

with open("막전.json", 'r', encoding='utf-8') as f:
    data = f.read()
load_data = json.loads(data)

for i in storeList :
    print("{0}의 데이터 타입 = {1}".format(i,type(load_data[0][i])))

print("데이터의 개수는 {} 입니다.".format(len(load_data)))


#%%
