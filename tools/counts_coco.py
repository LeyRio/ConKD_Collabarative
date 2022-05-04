import json
import os
file_path = os.path.join(os.path.dirname(os.path.abspath("__file__")),'appendix/coco/annotations/instances_val2017.json')
val=json.load(open(file_path, 'r'))
val.keys()