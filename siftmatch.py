from siftdetector import detect_keypoints   
import numpy as np
import cv2
import itertools

def match_template(imagename, templatename , threshold, cutoff):
    
    img = cv2.imread(imagename)
    template = cv2.imread(templatename)

    [kpi, di] = detect_keypoints(imagename, threshold)
    [kpt, dt] = detect_keypoints(templatename, threshold)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(np.asarray(di, np.float32), flann_params)
    idx, dist = flann.knnSearch(np.asarray(dt, np.float32), 2, params={})
    del flann

    dist.tolist()
    idx.tolist()
    kpi_cut = []
    kpt_cut = []
    count=0
    for i, dis in itertools.izip(idx, dist):
        if dis[0] < 0.6*dis[1]:
            kpi_cut.append(kpi[i[0]])
            kpt_cut.append(kpt[count])
        # elif dis[1] < 0.6*dis[0]:
        #     print("hula")
        #     kpi_cut.append(kpi[i[1]])
        #     kpt_cut.append(kpt[count])
        count+=1
    print "matched points", len(kpi_cut)
    print "dist ", np.max(dist)

    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h1 - h2) / 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[:h2,:w2] = template
    newimg[:h1, w2:w1+w2] = img

    for i in range(len(kpi_cut)):
        pt_a = (int(kpt_cut[i][1]), int(kpt_cut[i][0]))
        pt_b = (int(kpi_cut[i][1] + w2), int(kpi_cut[i][0]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))

    cv2.imwrite('matches.jpg', newimg)


def match_template1(imagename, templatename, threshold, cutoff):
    
    img = cv2.imread(imagename)
    template = cv2.imread(templatename)

    [kpi, di] = detect_keypoints(imagename, threshold)
    [kpt, dt] = detect_keypoints(templatename, threshold)
    drawFeatures(imagename, kpi, "keypoints.jpg")
    drawFeatures(templatename, kpt, "keypoints1.jpg")
    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h1 - h2) / 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[:h2,:w2] = template
    newimg[:h1, w2:w1+w2] = img
    matches=0
    for i in range(len(kpi)):
        pt_b = (int(kpi[i,1] + w2), int(kpi[i,0]))
        angles = np.arccos(np.matmul(di[i,:], dt.T))
        indices = range(len(dt))
        indices.sort(key=lambda i: angles[i])
        if(angles[indices[0]]<0.6*angles[indices[1]]):
            pt_a = (int(kpt[indices[0],1]), int(kpt[indices[0],0]))
            cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
            matches+=1
    print "matched points ", matches
    cv2.imwrite('matches.jpg', newimg)

def drawFeatures(imagename, keypoints, filename):
    newimg = cv2.imread(imagename)
    r=5.0
    for i in range(len(keypoints)):
        newimg = cv2.arrowedLine(newimg, (int(keypoints[i,1]), int(keypoints[i,0])), (int(keypoints[i,1]+r*keypoints[i,4]*np.sin(np.pi/18*keypoints[i,3])), int(keypoints[i,0] +r*keypoints[i,4]*np.cos(np.pi/18*keypoints[i,3]))), 255, tipLength=0.1)
    cv2.imwrite(filename, newimg)