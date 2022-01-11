"""
병합하는 코드인데 안쓰고 3차 전처리에서 합쳐서 만듬
아래 코드는 안쓰는 코드임
"""

import json
import glob

result = []

for f in glob.glob("2차 전처리/*.json"):
    with open(f, "rb") as infile:
        result.append(json.load(infile))

with open("merged_file.json", "w") as outfile:
     json.dump(result, outfile,indent=4)