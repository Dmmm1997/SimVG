import json
from collections import defaultdict

merge_annsfile = [
    "data/seqtr_type/annotations/refcoco-unc/instances.json",
    "data/seqtr_type/annotations/refcocoplus-unc/instances.json",
    "data/seqtr_type/annotations/refcocog-umd/instances.json",
    # "data/seqtr_type/annotations/refcocog-google/instances.json",
]

total_anns = defaultdict(list)
for annsfile in merge_annsfile:
    name = annsfile.split("/")[-2]
    single_ann = json.load(open(annsfile, 'r'))
    total_anns["train"].extend(single_ann.pop("train"))
    for key in single_ann.keys():
        total_anns["{}_{}".format(key, name.replace("-","_"))].extend(single_ann[key])

target_annsfile = "data/seqtr_type/annotations/mixed-seg/instances_nogoogle.json"
with open(target_annsfile, 'w') as f:
    json.dump(total_anns, f)
    
