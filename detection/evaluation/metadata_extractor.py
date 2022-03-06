from PIL import Image, ExifTags
import glob
from collections import defaultdict,Counter
import json
import math

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))



counter_dict = Counter()

labels=[
    'Model',
    ]

def roundup(x):
    return int(math.ceil(x / 1000.0)) * 1000
pths = glob.glob('d:/images/*')
print(len(pths))
for pth in pths:
    img = Image.open(pth)
    (width, height) = img.size
    try:
        values = hashabledict()
        (roundup(width), roundup(height))
        exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() }
        #for l in labels:
        #    values[l]=exif[l]

        dpi = exif['XResolution'][0] / exif['XResolution'][1]
        counter_dict[dpi]+=1
    except Exception as e:

        counter_dict["UNKNWON"] += 1

for k, v in counter_dict.items():
    print(str(v)+' : ' + str(k))



