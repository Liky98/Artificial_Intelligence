import json
import pandas as pd
import gc # 메모리 대비
import printr # 자료 찍어보기

path = "dblpv13.json"
#DBLP_dataset = json.load(open(path, 'r', encoding='UTF-8'))


print(gc.collect())
#%%
def yield_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f: #파일을 열어서 읽기 : r
        for line in f: #파일을 열어서 한 줄씩 읽어 줌
            data = json.loads(line) # json파일을 dictionary 형태로 읽어서 data에 넣어준다.
            yield data["id"],data["content"] #yield로 id, contet 칼럼만 반환해 주는데 generator로 만든다.

data = y


#%%
N = 30
with open(path, encoding='utf-8') as myfile:
    head = [next(myfile) for x in range(N)]

for l in head:
    print(l)

#%%
with open("input.txt") as f:
    data = f.readlines()
    for line in data:
        process(line)

#%%
####################################################
##  Type2: file.open and mmap.readline
##
##  파일을 메모리에 올린 후 읽기 (대용량 파일에 적합)
##  ==> 확실히 빨라진다 (8.1G 파일 기준, 처리시간 포함해 28분이 13분으로 줄어듬)

import mmap

def read_docs_type2(input_path: path, fname: str):
    rfile = input_path / fname
    docs: List[Document] = list()
    with open(rfile, "r+b", encoding='utf-8') as f:
        # length=0 mean 'whole of file'
        map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        for line in iter(map_file.readline, b""):   # read bytes
            doc = Document.from_json(line)
            docs.append( doc )
        map_file.close()        # must!!
    return docs

# ==> read_docs_type2: 2400909 docs, elapsed 784.430 sec (13분)     # mmap.readline
