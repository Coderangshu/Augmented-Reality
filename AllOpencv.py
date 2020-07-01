import cv2
import numpy as np

detection=False
frameCounter=0

cap = cv2.VideoCapture(0)
myVid = cv2.VideoCapture("video.mp4")
imgTarget = cv2.imread("Target.jpg",cv2.IMREAD_GRAYSCALE)

success, imgVideo = myVid.read()
hT, wT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)

#imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)

while True:
    success, imgWebcam = cap.read()
    imgWebcamGray=cv2.cvtColor(imgWebcam,cv2.COLOR_BGR2GRAY)
    img2,imgWarp,imgAug=imgWebcam.copy(),imgWebcam.copy(),imgWebcam.copy()
    maskNew = np.zeros((imgAug.shape[0], imgAug.shape[1]), np.uint8)  # create blank image of Augmentation size
    maskInv = maskNew
    kp2, des2 = orb.detectAndCompute(imgWebcamGray, None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    if detection==False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures=cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)

    if len(good)>25:
        detection=True
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        print(matrix)

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3, cv2.LINE_AA)  # draw polylines

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (img2.shape[1], img2.shape[0]))

        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))  # fill the detected area with white pixels to get mask
        maskInv = cv2.bitwise_not(maskNew)  # get inverse mask
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)  # make augmentation area black in final image
        imgAug = cv2.bitwise_or(imgWarp, imgAug)  # add final image with warped image


    cv2.imshow('imgAug',imgAug)
    #cv2.imshow('imgWarp',imgWarp)
    #cv2.imshow('img2',img2)
    #cv2.imshow('ImgFeatures', imgFeatures)
    #cv2.imshow('ImgTarget', imgTarget)
    #cv2.imshow('WCTarget', imgWebcam)
    #cv2.imshow('VidTarget', imgVideo)
    cv2.waitKey(1)
    frameCounter+=1