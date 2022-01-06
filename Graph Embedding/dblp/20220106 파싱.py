import json
import pandas as pd
import gc
import pickle
import re

count = 0

def processing() :
    global load_data
    global count

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


""" NumberInt값 대체하고 json 파일 열기 """

path = "C:/Users/LeeKihoon/Desktop/새 폴더/split"

for i in range(17) :
    with open(path+str(i+5)+".json", 'r', encoding='utf-8') as f:
        data = f.read()
    fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data) #NumberInt(0) 변환하기
    load_data = json.loads(fixed_data)
    print("parse_json result: %s" % type(data))
    try:
        processing()
    except :
        print("전처리끝")

    with open("처리완료 "+str(i+5)+"번째.json", 'w') as f:
        json.dump(load_data, f, indent=4)

