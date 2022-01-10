import json
storeList = ["_id", "title", "year", "keywords", "fos", "references", "n_citation", "abstract", "authors._id",
     "authors.name", "authors.org", "authors.orgid", "venue.sid", "venue.raw"]

with open("막전.json", 'r', encoding='utf-8') as f:
    data = f.read()
load_data = json.loads(data)

for i in storeList :
    print("{0}의 데이터 타입 = {1}".format(i,type(load_data[0][i])))

print("데이터의 개수는 {} 입니다.".format(len(load_data)))
