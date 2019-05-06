from siftmatch import *
match_template1("data/indiagater40.jpg","data/indiagate.jpg" , 5, 100)


# from siftdetector import detect_keypoints
# import numpy as np
# from scipy import ndimage
# import cv2
# import itertools
# imagename =  "data/indiagate.jpg"
# [keypoints, descriptors] = detect_keypoints(imagename,5)
# drawFeatures(imagename, keypoints, "keypoints.jpg")
# print "--------------------"
# imagename =  "data/indiagater.jpg"
# [keypoints, descriptors] = detect_keypoints(imagename, 5)
# drawFeatures(imagename, keypoints, "keypoints.jpg")



# from siftdetector import detect_keypoints
# import numpy as np
# import cv2
# import itertools
# imagename="data/indiagate.jpg"
# templatename="data/indiagater.jpg"
# threshold=5
# img = cv2.imread(imagename)
# template = cv2.imread(templatename)
# [kpi, di] = detect_keypoints(imagename, threshold)