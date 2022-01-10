import json
import glob

result = []

for f in glob.glob("2차 전처리/*.json"):
    with open(f, "rb") as infile:
        result.append(json.load(infile))

with open("merged_file.json", "w") as outfile:
     json.dump(result, outfile,indent=4)