""" 전역함수 라이브러리 설정"""
import json
import pandas as pd
import gc 
import pickle
import re

count = 0
path = "C:/Users/LeeKihoon/Desktop/새 폴더/split2.json"
#%%
""" NumberInt값 대체하고 json 파일 열기 """

with open(path, 'r', encoding='utf-8') as f:
    data = f.read()
fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data) #NumberInt(0) 변환하기
load_data = json.loads(fixed_data)
#print(json.dumps(load_data, indent=4))
print("parse_json result: %s" % type(data))

#%%
""" 결측값 확인 함수 """
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

#%%
""" 각 딕셔너리 별로 결측 데이터 확인하기 """
count = 0
try :
    processing()
except :
    print('끝났습니다')
#%%
""" 키-벨류 값 확인 """
for key, value in load_data[1].items():
    print(key, " : ", value)

#%%
"""json파일로 저장"""
with open("첫번째.json", 'w') as f :
    json.dump(load_data, f, indent=4)