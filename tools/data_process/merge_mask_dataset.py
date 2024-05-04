import json
from collections import defaultdict

merge_annsfile = [
    "data/annotations/refcoco-unc/instances.json",
    "data/annotations/refcocoplus-unc/instances.json",
    "data/annotations/refcocog-umd/instances.json",
    "data/annotations/refcocog-google/instances.json",
]

total_anns = defaultdict(list)
for annsfile in merge_annsfile:
    name = annsfile.split("/")[-2]
    single_ann = json.load(open(annsfile, 'r'))
    total_anns["train"].extend(single_ann.pop("train"))
    for key in single_ann.keys():
        total_anns["{}_{}".format(key, name.replace("-","_"))].extend(single_ann[key])

target_annsfile = "data/annotations/mixed-seg/instances.json"
with open(target_annsfile, 'w') as f:
    json.dump(total_anns, f)
    
