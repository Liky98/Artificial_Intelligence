""" 1차 전처리를 거치고 필요없는 키값들 삭제하여 최종 분할 데이터셋 완성 """
import json
import pandas as pd
import gc
import pickle
import re

count = 0

del_list = ["page_start","page_end", "doc_type", "lang", "volume",
            "issue", "issn", "isbn", "doi", "pdf", "url", "indexed_abstract"]

def processing() :
    global load_data
    global count
    global del_list

    while 1 :
        for list in del_list :
            if list in load_data[count] :
                del load_data[count][list]
        count+=1

for i in range(17) :
    with open("1차 전처리/처리완료 {0}번째.json".format(i+1), 'r', encoding='utf-8') as f:
        data = f.read()
    fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data) #NumberInt(0) 변환하기
    load_data = json.loads(fixed_data)
    print("parse_json result: %s" % type(data))
    print("{0}번째 Json파일을 열었습니다.".format(i+1))

    try:
        processing()
    except IndexError :
        print("전처리끝")
    except :
        print("다른에러")

    with open("2차 전처리 {0}번째.json".format(i+1), 'w') as f:
        json.dump(load_data, f, indent=4)


#%% 테스트코드
# with open("test.json", 'r', encoding='utf-8') as f:
#     data = f.read()
# fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data) #NumberInt(0) 변환하기
# load_data = json.loads(fixed_data)
# print("parse_json result: %s" % type(data))
# print("{0}번째 Json파일을 열었습니다.".format(1))
# try:
#     processing()
# except IndexError:
#     print("전처리끝")
# except:
#     print("다른에러")
#
# with open("테스트 "+str(1)+"번째.json", 'w') as f:
#     json.dump(load_data, f, indent=4)
#%%
file = open("test.json","r",encoding='utf-8')
jsondata = json.load(file)

print(type(jsondata))
print(jsondata)