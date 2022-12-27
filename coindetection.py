from cv2 import cv2
import numpy as np
import math

def nothing(x):
    pass

cv2.namedWindow('threshold_sliders', flags = cv2.WINDOW_NORMAL)
cv2.createTrackbar('Canny thresh1', 'threshold_sliders', 70, 300, nothing)
cv2.createTrackbar('Canny thresh2', 'threshold_sliders', 200, 300, nothing)
cv2.createTrackbar('HoughLinesP accumulator thresh', 'threshold_sliders', 100, 300, nothing)
cv2.createTrackbar('HoughLinesP minLineLength thresh', 'threshold_sliders', 100, 300, nothing)
cv2.createTrackbar('HoughLinesP maxLineGap thresh', 'threshold_sliders', 4, 300, nothing)
cv2.createTrackbar('maxDist merge lines thresh', 'threshold_sliders', 15, 300, nothing)
cv2.createTrackbar('HoughCircles minCenterDist', 'threshold_sliders', 100, 300, nothing)
cv2.createTrackbar('HoughCircles Canny thresh', 'threshold_sliders', 100, 300, nothing)
cv2.createTrackbar('HoughCircles accumulator thresh', 'threshold_sliders', 30, 300, nothing)

while True:
    canny_param1 = cv2.getTrackbarPos('Canny thresh1', 'threshold_sliders')
    canny_param2 = cv2.getTrackbarPos('Canny thresh2', 'threshold_sliders')
    line_accumulator_thresh = cv2.getTrackbarPos('HoughLinesP accumulator thresh', 'threshold_sliders')
    minLength = cv2.getTrackbarPos('HoughLinesP minLineLength thresh', 'threshold_sliders')
    maxGap = cv2.getTrackbarPos('HoughLinesP maxLineGap thresh', 'threshold_sliders')
    maxDist_merge = cv2.getTrackbarPos('maxDist merge lines thresh', 'threshold_sliders')
    circles_mindist = cv2.getTrackbarPos('HoughCircles minCenterDist', 'threshold_sliders')
    circles_cannythresh = cv2.getTrackbarPos('HoughCircles Canny thresh', 'threshold_sliders')
    circles_accumulatorthresh = cv2.getTrackbarPos('HoughCircles accumulator thresh', 'threshold_sliders')
    try:
        img = cv2.imread('C:\\Users\\quntu\\Desktop\\coding\\coins.jpg')
        img = cv2.resize(img, (int(img.shape[1]*.3), int(img.shape[0]*.3)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel, iterations = 1)
        canny = cv2.Canny(gray, canny_param1, canny_param2, apertureSize=3)
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, line_accumulator_thresh, minLineLength=minLength, maxLineGap=maxGap)
        lines = lines.tolist()

        unnested_lines = []
        for subline in lines:
            for line in subline:
                unnested_lines.append(line[::-1])

        shownlines = []
        for line in unnested_lines:
            y2, x2, y1, x1 = line
            slope = (y2 - y1)/(x2-x1)
            yintercept = round((-slope*x1 + y1), 2)
            line.insert(0, yintercept)
            shownlines.append(line)

        shownlines = sorted(shownlines)

        index = 0
        final_lines = []
        for line in shownlines:
            if index == 0:
                index += 1
                continue
            if shownlines[index][0] - shownlines[index-1][0] < maxDist_merge: #cv2.getTrackbarPos('Max distance threshold to merge lines')
                index += 1
                continue
            final_lines.append(line)
            index += 1

        distances = []
        index = 0
        for coords in final_lines:
            cv2.line(img, (int(coords[4]), int(coords[3])), (int(coords[2]), int(coords[1])), (0, 255, 0), 2)
            if index == 0:
                index += 1
                continue
            dx = final_lines[index][2]-final_lines[index-1][2]
            dy = final_lines[index][1]-final_lines[index-1][1]
            distance_formula = math.sqrt((dx)**2 + (dy)**2)
            distance = round(math.sqrt((distance_formula)**2 - dx**2), 2)
            distances.append(distance)
            index += 1


        distances = np.array(distances)
        mean = np.mean(distances)
        std = np.std(distances)
        corrected_distances = []
        for distance in distances:
            if distance > mean - std and distance < mean + std:
                corrected_distances.append(distance)

        average = round(sum(corrected_distances)/len(corrected_distances), 2)

        gray = cv2.bilateralFilter(gray, 9, 85, 85)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, .9, circles_mindist, param1=circles_cannythresh, param2=circles_accumulatorthresh, minRadius=0, maxRadius=0)
        detected_circles = np.uint16(np.around(circles))

        coins = {
            'quarter':[],
            'nickel':[],
            'dime':[],
            'penny':[]
        }

        for x, y, r in detected_circles[0, :]:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            width = (2*r/average)*.34375
            if width > 0.955:
                cv2.putText(img, 'quarter', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                coins.setdefault('quarter', []).append(1)
                continue
            if width > 0.835:
                cv2.putText(img, 'nickel', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                coins.setdefault('nickel', []).append(1)
                continue
            if width > .75:
                cv2.putText(img, 'penny', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                coins.setdefault('penny', []).append(1)
                continue
            if width > .7:
                cv2.putText(img, 'dime', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                coins.setdefault('dime', []).append(1)
                continue

        total = len(coins['quarter'])*.25 + len(coins['nickel'])*.05 + len(coins['dime'])*.1 + len(coins['penny'])*0.01
        value = str(round(total, 2))

        cv2.putText(img, 'Value: ' + '$' + value, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
        cv2.putText(img, '# of quarters: ' + str(len(coins['quarter'])), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
        cv2.putText(img, '# of dimes: ' + str(len(coins['dime'])), (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        cv2.putText(img, '# of nickels: ' + str(len(coins['nickel'])), (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)       
        cv2.putText(img, '# of pennies: ' + str(len(coins['penny'])), (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)      

        cv2.imshow('thresh', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    except:
        print('Error: bad threshold values, no coins in image, or no wide-ruled notebook paper')
        break
cv2.destroyAllWindows()