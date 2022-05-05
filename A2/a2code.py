import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from cv2.xfeatures2d import matchGMS

# import matplotlib.pyplot as plt

def ques_1(ratio,ORB_par,dis):
    # r1_1 = cv2.imread('book_covers/Reference/001.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # q1_1 = cv2.imread('book_covers/Query/001.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #greyscale
    r1_1 = cv2.imread('book_covers/Reference/001.jpg')
    q1_1 = cv2.imread('book_covers/Query/001.jpg')
    r1_1 = cv2.cvtColor(r1_1, cv2.COLOR_BGR2GRAY)
    q1_1 = cv2.cvtColor(q1_1, cv2.COLOR_BGR2GRAY)

    # compute detector and descriptor
    if ORB_par == 0:
        orb = cv2.ORB_create()
    else:
        orb = cv2.ORB_create(ORB_par)
    k_p_1 = orb.detect(r1_1, None)
    k_p_2 = orb.detect(q1_1, None)

    # find the keypoints and descriptors with ORB
    k_p_1, des_1 = orb.compute(r1_1, k_p_1)
    k_p_2, des_2 = orb.compute(q1_1, k_p_2)

    # draw keypoints
    # output = cv2.drawKeypoints(r1_1, k_p_1, 0)
    # plt.imshow(output)
    # plt.show()

    # create BFMatcher object
    # bf = cv2.BFMatcher()
    bf = cv2.BFMatcher(dis)
    # Match descriptors.
    match_list = bf.knnMatch(des_1, des_2, k=2)
    # match_list = sorted(match_list, key = lambda x : x.distance , reverse= False)

    # Apply ratio test
    ratio = ratio
    good_match = []
    for m, n in match_list:
        if m.distance / n.distance < ratio:
            good_match.append(m)
    # print(good_match)
    # draw matches
    # output = cv2.drawMatchesKnn(r1_1,k_p_1,q1_1,k_p_2,good_match,None)
    print("num of match point",len(good_match))
    output = cv2.drawMatches(r1_1, k_p_1, q1_1, k_p_2, good_match, None)
    plt.imshow(output)
    plt.show()

def que_compare(que, ref, match_count,ORB_para):
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    que = cv2.cvtColor(que, cv2.COLOR_BGR2GRAY)
    if ORB_para == 0:
        orb = cv2.ORB_create()
    else:
        orb = cv2.ORB_create(ORB_para)
    k_p_r = orb.detect(ref, None)
    k_p_q = orb.detect(que, None)
    k_p_r, des_r = orb.compute(ref, k_p_r)
    k_p_q, des_q = orb.compute(que, k_p_q)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_r, des_q, k=2)
    rat = 0.8
    take = []
    for m, n in matches:
        if m.distance / n.distance < rat:
            take.append(m)
    # print(len(take))
    # if len(take) < 4:  # threshold
    #     match_count.append(0)
    #     return match_count
    r_ptr = np.float32([k_p_r[m.queryIdx].pt for m in take]).reshape(-1, 1, 2)
    q_ptr = np.float32([k_p_q[m.trainIdx].pt for m in take]).reshape(-1, 1, 2)
    if len(r_ptr) < 4 or len(q_ptr) < 4:
        match_count.append(0)
        return match_count
    homography, mask = cv2.findHomography(r_ptr, q_ptr, cv2.RANSAC, 5)
    sum = 0
    for i in mask:
        if i == 1:
            sum += 1
    match_count.append(sum)
    return match_count


def best_match_ref(data_base,extra_data,que_num, img_accept,ORB_para,threshold):
    # img_accept = 1  # number of images you going to receive in output
    # que_num = 1  # the image number， for example: 001.jpg  -> 1     021.jpg -> 21

    if data_base == "book_covers":
        d_b_n = 101
    elif data_base == "landmarks":
        d_b_n = 100
    else:  # museum_painting
        d_b_n = 91

    match_count = []
    num = str(que_num)
    f_num = num.zfill(3)
    if que_num <= d_b_n:
        que = cv2.imread(data_base+'/Query/' + f_num + '.jpg')
    else:
        if extra_data == '':
            return 0;
        que_num -= d_b_n;
        print(extra_data+'/Query/' + (str(que_num)).zfill(3) + '.jpg')
        que = cv2.imread(extra_data+'/Query/' + (str(que_num)).zfill(3) + '.jpg')
    # for loop taking ref
    for i in range(1, d_b_n+1):
        num = str(i)
        f_num = num.zfill(3)
        ref = cv2.imread(data_base+'/Reference/' + f_num + '.jpg')
        match_count = que_compare(que, ref, match_count,ORB_para)



    match_count = np.array(match_count)

    top_ten = (-match_count).argsort()[:img_accept]
    # print(top_ten)
    # print(que_num-1,": top_ten: ",top_ten, " match points num:",match_count[top_ten])
    # print(que_num,": marks  : ", )

    if match_count[top_ten[0]] < threshold:
        print("not in dataset  ",que_num)
        return 0
    else:

        found = 1  # if our que and match_ref is the same number, we have a true [ref001.jpg -> que001.jpg  to true]
        for j in top_ten:
            # print(j)
            if j + 1 == que_num:
                # if j != top_ten[0]:
                #     # print()
                found = 2
                # print("Match:\nref " + (str(j + 1)).zfill(3) + ".jpg\nque " + (str(que_num)).zfill(
                #     3) + ".jpg\nMatch point :" + str(match_count[top_ten[j]]) + "\n")
        return found

def get_thres(data_base,ORB_para):
    if data_base == "book_covers":
        if ORB_para == 10000:
            threshold = 30
        elif ORB_para == 1000:
            threshold = 14
        else:
            threshold = 12
    else :
    # if data_base == "landmarks":
        threshold = 15
    return threshold


def match_accuracy(data_base,img_accept,ORB_para):
    match_num = 0
    total = 101
    threshold = get_thres(data_base,ORB_para)
    # for i in range(1, 102):
    for i in range(1, total+1):
        if best_match_ref(data_base,'',i, img_accept,ORB_para,threshold) == 2:
            match_num += 1
    # return float(match_num/total)
    return match_num/total

def match_accuracy_not_find(data_base,extra_data,img_accept,num_extra,ORB_para):
    if data_base == "book_covers":
        d_b_n =101
    elif data_base == "landmarks":
        d_b_n = 100
    else : #museum_painting
        d_b_n = 91
    match_num = 0
    no_match = 0
    threshold = get_thres(data_base, ORB_para)
    total = d_b_n+num_extra
    for i in range(1, total+1):
        result = best_match_ref(data_base,extra_data,i, img_accept,ORB_para,threshold)
        if result == 2:
            match_num += 1
        elif result == 0 and i > d_b_n:
            print("found 1 not match ",i)
            no_match += 1


    # return float(match_num/total)
    return (match_num+no_match)/total

#q3

def ratio_match(r_num):
    q_num = r_num
    # f_num = (str(1)).zfill(3)
    ref = cv2.imread('book_covers/Reference/' + (str(r_num)).zfill(3) + '.jpg')
    que = cv2.imread('book_covers/Query/' + (str(q_num)).zfill(3) + '.jpg')

    orb = cv2.ORB_create(10000)
    # orb = cv2.ORB_create()
    orb.setFastThreshold(0)

    k_p_r, des_r = orb.detectAndCompute(ref, None)
    k_p_q, des_q = orb.detectAndCompute(que, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    match_points = bf.match(des_r, des_q)
    match_gms = matchGMS(ref.shape[:2], que.shape[:2], k_p_r, k_p_q, match_points, withScale=False, withRotation=False,
             thresholdFactor = 6)
    print("match num: ",len(match_gms))

    # match_gms = matchGMS(ref.shape[:2], que.shape[:2], k_p_r, k_p_q, match_points, withScale=False, withRotation=False,
    #          thresholdFactor = 6)
    plt.imshow(cv2.drawMatches(ref,k_p_r,que,k_p_q,match_gms,None))
    plt.show()
    return



def SIFT_Match(data_base,extra_data,img_accept,num_extra,ORB_para):
    if data_base == "book_covers":
        d_b_n =101
    elif data_base == "landmarks":
        d_b_n = 100
    else : #museum_painting
        d_b_n = 91
    match_num = 0
    no_match = 0
    threshold = get_thres(data_base, ORB_para)
    total = d_b_n+num_extra
    for i in range(1, total+1):
        match_count = []
        num = str(i)
        f_num = num.zfill(3)
        if i <= d_b_n:
            que = cv2.imread(data_base + '/Query/' + f_num + '.jpg')
        else:
            if extra_data == '':
                return 0;
            i -= d_b_n;
            print(extra_data + '/Query/' + (str(i)).zfill(3) + '.jpg')
            que = cv2.imread(extra_data + '/Query/' + (str(i)).zfill(3) + '.jpg')
        # for loop taking ref
        for a in range(1, d_b_n + 1):
            num = str(a)
            f_num = num.zfill(3)
            ref = cv2.imread(data_base + '/Reference/' + f_num + '.jpg')
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            que = cv2.cvtColor(que, cv2.COLOR_BGR2GRAY)
            if ORB_para == 0:
                orb = cv2.ORB_create()
            else:
                orb = cv2.SIFT_create(ORB_para)
            k_p_r = orb.detect(ref, None)
            k_p_q = orb.detect(que, None)
            k_p_r, des_r = orb.compute(ref, k_p_r)
            k_p_q, des_q = orb.compute(que, k_p_q)
            bf = cv2.BFMatcher(5)
            matches = bf.knnMatch(des_r, des_q, k=2)
            rat = 0.8
            take = []
            for m, n in matches:
                if m.distance / n.distance < rat:
                    take.append(m)

            r_ptr = np.float32([k_p_r[m.queryIdx].pt for m in take]).reshape(-1, 1, 2)
            q_ptr = np.float32([k_p_q[m.trainIdx].pt for m in take]).reshape(-1, 1, 2)
            if len(r_ptr) < 4 or len(q_ptr) < 4:
                match_count.append(0)
                return match_count
            homography, mask = cv2.findHomography(r_ptr, q_ptr, cv2.RANSAC, 5)
            sum = 0
            for b in mask:
                if b == 1:
                    sum += 1
            match_count.append(sum)

        match_count = np.array(match_count)

        top_ten = (-match_count).argsort()[:img_accept]
        if match_count[top_ten[0]] < threshold:
            print("not in dataset  ", i)
            return 0
        else:
            found = 1  # if our que and match_ref is the same number, we have a true [ref001.jpg -> que001.jpg  to true]
            for j in top_ten:
                if j + 1 == i:
                    found = 2
        result = found
        if result == 2:
            match_num += 1
        elif result == 0 and i > d_b_n:
            print("found 1 not match ",i)
            no_match += 1
    return (match_num+no_match)/total