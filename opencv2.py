from __future__ import print_function
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from cv2 import cv2 as cv2
from PIL import Image
import imagehash
from moviepy.editor import VideoFileClip
import sys


def alignImages(img1, img2):
    orb = cv2.ORB_create(5000)
    img1 = img1[150:850,:]
    img2= img2[150:850,:]
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    #bf匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
    good = [m for (m,n) in matches if m.distance < 0.35*n.distance]

    min_match_count=5
    if len(good)>min_match_count:
        #匹配点
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#         change = (((src_pts-dst_pts)[int(len(src_pts)/2)])[0])[1]
        x_change=(src_pts-dst_pts).sum(axis=0)[0][1]
        y_change=(src_pts-dst_pts).sum(axis=0)[0][0]
        return x_change,y_change


        # #找到变换矩阵m
        # m,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5,0)
        # matchmask=mask.ravel().tolist()
        # h,w=img1.shape
        # pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        # dst=cv2.perspectiveTransform(pts,m)
        # cv2.polylines(img2,[np.int32(dst)],True,255,10,cv2.LINE_AA)
    else:
        print('not enough matches are found  -- %d/%d',(len(good),min_match_count))
        return 9999


def save_image(image,addr,num):
    address = addr + str(num)+ '.jpg'
    cv2.imwrite(address,image)


def isSimilar(img1, img2):

	# OpenCV图片转换为PIL image
	img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)[150:850,])
	img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)[150:850,])

	# 通过imagehash获取两个图片的平均hash值
	n0 = imagehash.average_hash(img1)
	n1 = imagehash.average_hash(img2)

	# hash值最小相差多少则判断为不相似，可以根据需要自定义
	cutoff = 7

	# print(f'目标帧hash平均值：{n0}')
	# print(f'后帧hash平均值：  {n1}')
	print(f'hash差值：       {n0-n1}')

	# flag = True
	# if n0 - n1 < cutoff:
	# 	print('相似')
	# else:
	# 	flag = False
	# 	print('不相似')

	return n0-n1


if __name__ == '__main__':

  # # Read reference image
  # refFilename = "D:/Alibaba customer behaviour/1.png"
  # print("Reading reference image : ", refFilename)
  # imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  #
  # # Read image to be aligned
  # imFilename = "D:/Alibaba customer behaviour/2.png"
  # print("Reading image to align : ", imFilename);
  # im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
  #
  # print("Aligning images ...")
  # # Registered image will be resotred in imReg.
  # # The estimated homography will be stored in h.
  #
  # #imReg, h = alignImages(imReference,im)
  # alignImages(imReference,im)

    videoCapture = cv2.VideoCapture('C:/Users/11488/Desktop/zju/taobao_test1.mp4')
  # 通过摄像头的方式
  # videoCapture=cv2.VideoCapture(1)

  #读帧
    up=0
    down=0
    left=0
    right=0
  # fpslst=[]
  # scorelst=[]
    new_scene=[]
  #计算相机的帧速率FPS
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    print(fps)
  # 得到第一帧
    success, frame_old = videoCapture.read()
    i = 0

    timeF =15 #可以自己调节
    j=0
    save_image(frame_old,'C:/Users/11488/Desktop/zju/test/',j)
    print('save image:',i)

    while success :
        i = i + 1
        success, frame_new = videoCapture.read()
        if success==False:
            print('vedio detection finish')
            sys.exit()

    # if (i==fps/2):
    #   k=00
    #   save_image(frame_new,'D:/Alibaba customer behaviour/test/',k)
        if (i % timeF == 0):
            j = j + 1
            save_image(frame_new,'C:/Users/11488/Desktop/zju/test/',j)
            print('save image:',i)
      # new_scene.append(isSimilar(frame_new,frame_old))
      # print(new_scene)
      # if isSimilar(frame_new,frame_old)==False:
      #     new_scene.append(i)
      # else:
      #     continue


  #     fpslst.append(j)
  #     print(alignImages(frame_old,frame_new))
  #     grayA = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)[150:850,:]
  #     grayB = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)[150:850,:]
  #     (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
  #     scorelst.append(score)
  #     #print('ssim:',score)
            
        if alignImages(frame_old,frame_new)==9999:
            continue
        else:
            if alignImages(frame_old,frame_new)==(0,0):
                continue
            else:
                
                if alignImages(frame_old,frame_new)[0]>0:
                    up+=1
            
                elif alignImages(frame_old,frame_new)[0]<0:
                    down+=1
                elif alignImages(frame_old,frame_new)[1]>50:
                    right+=1
                elif alignImages(frame_old,frame_new)[1]<-50:
                    left+=1

            frame_old=frame_new
            print('up='+str(up)+'s','down='+str(down)+'s','left='+str(left)+'s','right='+str(right)+'s')
  # print(scorelst)
  # print(scorelst.index(min(scorelst)),scorelst.index(max(scorelst)))
  # plt.plot(fpslst, scorelst)
  # plt.show()
  # src = cv2.imread('D:/Alibaba customer behaviour/test/0.jpg')
  # img = cv2.imread('D:/Alibaba customer behaviour/test/1.jpg')
  # grayA = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  # grayB = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 计算两个灰度图像之间的结构相似度
  # (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
  # diff = (diff * 255).astype("uint8")
  # cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
  # cv2.imshow("diff", diff)
  # print("SSIM:{}".format(score))
