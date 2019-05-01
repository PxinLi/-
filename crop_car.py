import numpy as np
import glob
import xml.dom.minidom as xmldom
import cv2
import os
from PIL import Image

def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    print(eles.tagName)
    xmin = eles.getElementsByTagName("xmin")[0].firstChild.data
    xmax = eles.getElementsByTagName("xmax")[0].firstChild.data
    ymin = eles.getElementsByTagName("ymin")[0].firstChild.data
    ymax = eles.getElementsByTagName("ymax")[0].firstChild.data
    print(xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax


img_file = glob.glob('./car/*.jpg')
box_file = glob.glob('./car/*.xml')

for (img_name, box_name) in zip(img_file, box_file):
    print(img_name)
    xmin, xmax, ymin, ymax = parse_xml(box_name)
    img = Image.open(img_name)
    cropped = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

    if not os.path.exists('./crop_plate'):
        os.mkdir('./crop_plate')
    img_name_pre = img_name.split("\\")[1]
    cropped.save("./crop_plate/"+img_name_pre)

exit(0)