import json
def json_open(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    load_data = json.loads(data)
    return load_data
