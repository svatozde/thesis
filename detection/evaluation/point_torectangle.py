import glob
import os
import json

OUT_PATH = 'c:/skola/MODIFIED/'

add = 16
for pth in glob.glob('d:/labeled_last/*.json'):
    print(pth)
    labels = json.load(open(pth,))
    file_name = os.path.basename(pth)
    for shp in  labels["shapes"]:
        if "intersection" == shp["label"]:
            shp["label"] = "seam_connection"

        if "junction" == shp["label"] or "bot" == shp["label"] or "top" == shp["label"]:
            if "point" == shp["shape_type"]:
                shp["shape_type"] = "rectangle"
                point = shp["points"][0]
                x = float(point[0])
                y = float(point[1])
                point_1 = [x - add, y - add]
                point_2 = [x + add, y + add]
                shp["points"] = [point_1,point_2]



    outF = open(OUT_PATH + file_name, "w")
    json.dump(labels, outF, indent=2)
    outF.close()