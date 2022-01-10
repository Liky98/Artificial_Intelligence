""" 필수 키값이 전부 존재하는 데이터만 남기는 코드"""
import json
import pandas as pd
import gc
import pickle
import re

path = "C:/Users/LeeKihoon/Desktop/새 폴더/split"
count = 0
find_list = ["_id", "title", "authors", "venue", "year", "keywords",
             "fos", "references", "n_citation", "publisher", "abstract"]

def processing() :
    global load_data
    global count
    global del_list

    while 1 :
        for list in find_list :
            if list not in load_data[count] :
                del load_data[count][list]
        count+=1

#NumberInt값 대체하고 json 파일 열기
for i in range(17) :
    with open(path+str(i+1)+".json", 'r', encoding='utf-8') as f:
        data = f.read()
    fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data) #NumberInt(0) 변환하기
    load_data = json.loads(fixed_data)
    print("parse_json result: %s" % type(data))

    try:
        processing()
    except :
        print("전처리끝")

    with open("처리완료 "+str(i+1)+"번째.json", 'w') as f:
        json.dump(load_data, f, indent=4)

# The End

"""
        
    while 1 :
        if "_id" not in load_data[count]:
            del load_data[count] #딕셔너리 자체를 삭제
            count+=1
            continue
        elif "title" not in load_data[count] :
            del load_data[count]
            count+=1
            continue
        elif "authors" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "venue" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "year" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "keywords" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "fos" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "references" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "n_citation" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "publisher" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        elif "abstract" not in load_data[count]:
            del load_data[count]
            count += 1
            continue
        else :
            count +=1
            continue
"""