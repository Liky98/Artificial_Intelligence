import json
import pandas as pd
import gc 
import pickle
import re

path = "test.json"

with open(path, 'r', encoding='utf-8') as f:
    data = f.read()

fixed_data = re.sub(r"NumberInt\((\d+)\)", r"\1", data) #NumberInt(0) 변환하기

load_data = json.loads(fixed_data)

print(json.dumps(load_data, indent=4))
print("parse_json result: %s" % type(data))

#%%
load_data.keys()

#%% json파일로 저장
with open("storeData.json", "w") as f :
    pickle.dump(data, f)