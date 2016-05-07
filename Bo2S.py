# script is based on:
# https://github.com/ibininja/upload_file_python

import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory, jsonify

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.path as mplPath
import os.path
from copy import deepcopy
from skimage import img_as_ubyte
from math import atan2
import json
import time

# ========================================================== #
# Functions:
# triangleColor
def triangleColor(img,triangle):
    mask = np.zeros(img.shape,np.uint8)
    cv2.fillConvexPoly(mask,np.int32([triangle]),[1,1,1])
    mask = mask[:,:,1]
    mean_val = cv2.mean(img,mask = mask)
    threshold = 125
    return np.mean(mean_val[0:3])<threshold

# sortCornersRectangle
def sortCornersRectangle(corners):
    result = [] 
    xsorted = sorted(corners, key = lambda xs: xs[0])
    ysorted = sorted(corners, key = lambda ys: ys[1])
    def comp(p1, p2):
        if(p1[0] == p2[0] and p1[1] == p2[1]):
            return True
        else:
            return False

    if(comp(xsorted[0],ysorted[0]) or comp(xsorted[0], ysorted[1])):
        result.append(xsorted[0])
    else:
        result.append(xsorted[1])
    if(comp(xsorted[0], ysorted[2]) or comp(xsorted[0], ysorted[3])):
        result.append(xsorted[0])
    else:
        result.append(xsorted[1])
    if(comp(xsorted[2], ysorted[2]) or comp(xsorted[2], ysorted[3])):
        result.append(xsorted[2])
    else:
        result.append(xsorted[3])
    if(comp(xsorted[2], ysorted[0]) or comp(xsorted[2], ysorted[1])):
        result.append(xsorted[2])
    else:
        result.append(xsorted[3])
    return result

# sortCornersTriangle
def sortCornersTriangle(corners):   
    # sorting by polar angle 
    result = np.zeros((3,2))
    b = [np.mean(corners[:,0]), np.mean(corners[:,1])]
    a = [0,0,0]
    for i in range(0,3):
        a[i] = atan2(corners[i,1]-b[1],corners[i,0]-b[0])
    idx = np.argsort(a)
    for i in range(0,3):
        result[i,0] = corners[idx[i],0]
        result[i,1] = corners[idx[i],1]
    return result

# distance
def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# angles divided by PI    
def anglesPi(triangle):
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]
    a = dist(B,C)
    b = dist(A,C)
    c = dist(B,A)
    alpha = np.arccos((c**2+b**2-a**2)/(2*c*b))
    beta  = np.arccos((c**2+a**2-b**2)/(2*c*a))
    gamma = np.arccos((a**2+b**2-c**2)/(2*a*b))    
    #print A,B,C
    #print a,b,c
    #print alpha/np.pi, beta/np.pi, gamma/np.pi
    return [alpha/np.pi, beta/np.pi, gamma/np.pi]

# triangle number    
def triangleToNumber(image,triangle):
    # detect color
    c = triangleColor(image,triangle)
    ang = anglesPi(triangle)
    maxAng = max(ang)
    maxAngIdx = ang.index(maxAng)
    
    if(abs(triangle[maxAngIdx][0]-triangle[(maxAngIdx + 1) % 3][0]) 
    - abs(triangle[maxAngIdx][0]-triangle[(maxAngIdx + 2) % 3][0]) > 0):
        simxIdx = (maxAngIdx + 2) % 3
        difxIdx = (maxAngIdx + 1) % 3
    else:
        simxIdx = (maxAngIdx + 1) % 3
        difxIdx = (maxAngIdx + 2) % 3
    
    if(abs(triangle[maxAngIdx][1]-triangle[(maxAngIdx + 1) % 3][1]) 
    - abs(triangle[maxAngIdx][1]-triangle[(maxAngIdx + 2) % 3][1]) > 0):
        simyIdx = (maxAngIdx + 2) % 3
        difyIdx = (maxAngIdx + 1) % 3
    else:
        simyIdx = (maxAngIdx + 1) % 3
        difyIdx = (maxAngIdx + 2) % 3
        
    
    simx = triangle[simxIdx][0]
    difx = triangle[difxIdx][0]
    simy = triangle[simyIdx][1]
    dify = triangle[difyIdx][1]
    
    if(difx > simx):
        if(dify > simy):
            num = 0
        else:
            num = 1
    else:
        if(dify > simy):
            num = 2
        else:
            num = 3	
    return num*2+c            
    
# calculate triangle barycentrum   
def triangleCenter(triangle):
    x = float(triangle[0][0] + triangle[1][0] + triangle[2][0])/3
    y = float(triangle[0][1] + triangle[1][1] + triangle[2][1])/3
    return [x,y]
    
# angle
# finds a cosine of angle between vectors
# from pt0->pt1 and from pt0->pt2
def angle(pt1, pt2, pt0):
    # calculate differences
    dx1 = (pt1[0][0] - pt0[0][0])
    dy1 = (pt1[0][1] - pt0[0][1])
    dx2 = (pt2[0][0] - pt0[0][0])
    dy2 = (pt2[0][1] - pt0[0][1])
    # return angle
    return (dx1*dx2 + dy1*dy2)/np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);

# inpolygon
# finds if point p is inside the polygon pol
def inpolygon(pol, p):
    # create  path
    polyPath = mplPath.Path(pol)
    return polyPath.contains_point((p[0], p[1]))

# finds if point is inside the square
def pointInSquare(point, square):
    xsorted = sorted(square, key = lambda xs: xs[0])
    ysorted = sorted(square, key = lambda ys: ys[1])
    if (point[0] > xsorted[0][0] and point[0] < xsorted[3][0] and point[1] > ysorted[0][1] and point[1] < ysorted[3][1]):
        return True
    else:
        return False

def trianglesToSquaresAndNumbers(triangles, squares):
    trianglesInSquaresSorted = []

    for square in squares:
        tmp = []
        for triangle in sorted(triangles, key = lambda xs: triangleCenter(xs)[0]):
            if(pointInSquare(triangleCenter(triangle),square)):
                tmp.append(triangleShapeToNumber(triangle))
                #print triangle, square
        trianglesInSquaresSorted.append(tmp)
    return trianglesInSquaresSorted
        
#result = trianglesToSquaresAndNumbers(triangles, squares)

def diag(square):
    maxDiag = 0
    for corner0 in square:
        for corner1 in square:
            if(dist(corner0, corner1) > maxDiag):
                maxDiag = dist(corner0, corner1)
    return maxDiag
    
def detectSmallSquares(squares, howManyAreas):
    howMany = (howManyAreas+1)*2
    detectedSquares = sorted(squares, key = lambda xs: diag(xs))
    detectedSquares = detectedSquares[:howMany]
    detectedSquares = sorted(detectedSquares, key = lambda xs: xs[0][1])
    
    detectedSquares[:howMany/2] = sorted(detectedSquares[:howMany/2], key = lambda xs: xs[0][0])
    detectedSquares[howMany/2:] = sorted(detectedSquares[howMany/2:], key = lambda xs: xs[0][0])
    #for sq in detectedSquares:
    #    print diag(sq), sq
    return detectedSquares

def squareCenter(square):
    x = float(square[0][0] + square[1][0] + square[2][0] + square[3][0])/4
    y = float(square[0][1] + square[1][1] + square[2][1] + square[3][1])/4
    return [x,y]
    
def areas(squares, howManyAreas):
    smallSquares = detectSmallSquares(squares, howManyAreas)
    #for square in smallSquares:
        #print squareCenter(square)
    ar = []
    for i in range(howManyAreas):
        ar.append([[squareCenter(smallSquares[i])[0],squareCenter(smallSquares[i])[1]],
                   [squareCenter(smallSquares[i+1+howManyAreas])[0],squareCenter(smallSquares[i+1+howManyAreas])[1]],
                   [squareCenter(smallSquares[i+2+howManyAreas])[0],squareCenter(smallSquares[i+2+howManyAreas])[1]],
                   [squareCenter(smallSquares[i+1])[0],squareCenter(smallSquares[i+1])[1]]
                  ])
    return(ar)
        
def pointNotInFigure(point, area):
    maxX = sorted(area, key = lambda xs: xs[0])[len(area)-1][0]
    minX = sorted(area, key = lambda xs: xs[0])[0][0]
    maxY = sorted(area, key = lambda xs: xs[1])[len(area)-1][1]
    minY = sorted(area, key = lambda xs: xs[1])[0][1]
    
    if(point[0] < minX or point[0] > maxX or point[1] < minY or point[1] > maxY):
        return True
        
    return False
    
        
def allPpointsInArea(figure, area):
    for point in figure:
        if(pointNotInFigure(point, area)):
            return False
    return True
        
def squaresInAreas(squares, areas):
    areaNumber = []
    for i in range(len(squares)):
        for j in range(len(areas)):
            if (allPpointsInArea(squares[i], areas[j])):
                #print i,j+1
                #print squares[i]
                #print areas[j]
                areaNumber.append([i,j+1])
    return areaNumber

# findSquaresAndTriangles
# returns sequence of squares detected on the image.
def findSquaresAndTriangles(img):
    thresh  = 80;
    N = 5;
    sqr = [];
    trg = [];
    #plt.imshow(img,cmap = 'gray')
    #plt.show()
    img = cv2.medianBlur(img,1)
    #plt.imshow(img,cmap = 'gray')
    #plt.show()

    # find squares in every color plane of the image
    for c in range(0, 3):
        #ch = [c, 0]
        #cv2.mixChannels(img, gray0, ch)
        gray0 = img[:,:,c]
        #plt.imshow(gray0)
        #plt.show()
        # try several threshold levels
        for l in range(0, N):
            # hack: use Canny instead of zero threshold level.
            # Canny helps to catch squares with gradient shading
            if (l == 0):
                # apply Canny. Take the upper threshold from slider
                # and set the lower to 0 (which forces edges merging)
                gray = cv2.Canny(gray0, 5, thresh, 5);
                #plt.imshow(gray,cmap = 'gray')
                #plt.show()
                # dilate canny output to remove potential
                # holes between edge segments
                gray = cv2.dilate(gray,None)
                #plt.imshow(gray,cmap = 'gray')
                #plt.show()
                #print gray
            else:
                # apply threshold if l!=0:
                # tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = img_as_ubyte(gray0 >= (l+1)*255/N) # conversion needed
                #plt.imshow(gray,cmap = 'gray')
                #plt.show()
            # find contours and store them all as a list
            ####findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            gray_copy = deepcopy(gray) # deep copy for test, you can delete it
            contours, hierarchy = cv2.findContours(gray_copy,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(img, contours, -1, (0,255,0), 3)
            #plt.imshow(img)
            #plt.show()
            ####vector<Point> approx;

            # test each contour
            imarea = np.size(img, 0)*np.size(img, 1)
            for i in range(0,len(contours)): 
                # approximate contour with accuracy proportional
                # to the contour perimeter
                ####approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
                approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i], True)*0.02, True)
                # instead of 0.02 try to use another number, see for example: http://opencvpython.blogspot.com/2012/06/contours-2-brotherhood.html
                # square contours should have 4 vertices after approximation
                # relatively large area (to filter out noisy contours)
                # and be convex.
                # Note: absolute value of an area is used because
                # area may be positive or negative - in accordance with the
                # contour orientation
                # last condition remove border of the image based on area comparison
                if ( len(approx) == 4 and np.fabs(cv2.contourArea(approx)) > 1000 and cv2.isContourConvex(approx) and np.fabs(cv2.contourArea(approx))/imarea<0.9):
                    maxCosine = 0;
                    for j in range(2,5):
                        # find the maximum cosine of the angle between joint edges
                        cosine = np.fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = max(maxCosine, cosine);

                    # if cosines of all angles are small
                    # (all angles are ~90 degree) then write quandrange
                    # vertices to resultant sequence
                    if( maxCosine < 0.3 ):
                        sqr.append(approx)
                # triangular contours should have 3 vertices after approximation
                # relatively large area (to filter out noisy contours), but smaller than in cse of squares
                # and be convex.
                elif ( len(approx) == 3 and np.fabs(cv2.contourArea(approx)) > 10 and cv2.isContourConvex(approx) ):
                    minCosine = 1
                    #print "============"
                    for j in range(0,3):
                        # find the maximum cosine of the angle between joint edges
                        cosine = np.fabs(angle(approx[j], approx[(j+1)%3], approx[(j+2)%3]));
                        #print cosine
                        minCosine = min(minCosine,cosine)
                    if(minCosine < 0.2):
                        trg.append(approx)
    # find nonunique squares
    # first calculate centroids of each square
    centroids = np.zeros((len(sqr),2))
    for i in range(0,len(sqr)):
        centroids[i,0] = np.mean(sqr[i][:,0,0])
        centroids[i,1] = np.mean(sqr[i][:,0,1])
        #print centroids[i,0], centroids[i,1]
    # check centroids inside coutours
    areInside = np.zeros((len(sqr), len(sqr)), dtype=bool)
    for i in range(0,len(sqr)):
        a1 = np.fabs(cv2.contourArea(sqr[i]))
        for j in range(0,len(sqr)):
            # check condition if only squares have similar area            
            a2 = np.fabs(cv2.contourArea(sqr[j]))
            if (a1/a2 > 0.5 and a2/a1 < 2):
                areInside[i,j] = inpolygon(sqr[i][:,0,:], centroids[j])          
            #print [i, j, areInside[i,j]]
    # calculate "mean contours"
    #print areInside
    #plt.imshow(areInside)
    #plt.show()
    usqr = []
    toOmit = []
    for i in range(0,len(sqr)):        
        if (i in toOmit):
            continue
        tmp = np.where(areInside[:,i])
        toOmit.extend(tmp[0])
        tmp = list(tmp[0])
        usqr.append(np.zeros((4,2)))
        for j in range(0,len(tmp)):
            toSort = sqr[tmp[j]][:,0,:]
            #usqr[len(usqr)-1] = usqr[len(usqr)-1] + toSort[toSort[:,1].argsort()]
            usqr[len(usqr)-1] = usqr[len(usqr)-1] + sortCornersRectangle(toSort)
        usqr[len(usqr)-1] = usqr[len(usqr)-1]/len(tmp)
        
    # find nonunique triangles
    # first calculate centroids of each triangle
    centroids = np.zeros((len(trg),2))
    for i in range(0,len(trg)):
        centroids[i,0] = np.mean(trg[i][:,0,0])
        centroids[i,1] = np.mean(trg[i][:,0,1])
    # check centroids inside coutours
    areInside = np.zeros((len(trg), len(trg)), dtype=bool)
    for i in range(0,len(trg)):
        a1 = np.fabs(cv2.contourArea(trg[i]))
        for j in range(0,len(trg)):
            # check condition if only squares have similar area            
            a2 = np.fabs(cv2.contourArea(trg[j]))
            if (a1/a2 > 0.5 and a2/a1 < 2):
                areInside[i,j] = inpolygon(trg[i][:,0,:], centroids[j])          
    # calculate "mean contours"
    utrg = []
    toOmit = []
    for i in range(0,len(trg)):        
        if (i in toOmit):
            continue
        tmp = np.where(areInside[:,i])
        toOmit.extend(tmp[0])
        tmp = list(tmp[0])
        if len(tmp)==0:
            continue
        utrg.append(np.zeros((3,2)))
        for j in range(0,len(tmp)):
            toSort = trg[tmp[j]][:,0,:]            
            utrg[len(utrg)-1] = utrg[len(utrg)-1] + sortCornersTriangle(toSort)            
        utrg[len(utrg)-1] = utrg[len(utrg)-1]/len(tmp)
    return {'squares':usqr, 'triangles':utrg}

# drawContours
# the function draws all the squares in the image
def drawContours(img, ctr):
    for i in range(0,len(ctr)):
        pts = np.int32([ctr[i]])
        #print pts
        cv2.polylines(img, pts, True, [255,0,0], 3)
    return img

def trianglesToSquaresAndNumbers(triangles, squares, img):
    trianglesInSquaresSorted = []

    for square in squares:
        tmp = []
        for triangle in sorted(triangles, key = lambda xs: triangleCenter(xs)[0]):
            if(pointInSquare(triangleCenter(triangle),square)):
                tmp.append(triangleToNumber(img,triangle))
                #print triangle, square
        trianglesInSquaresSorted.append(tmp)
    return trianglesInSquaresSorted

# calculate statistics of each contour
def calculateStats(img,sqrs):
    stats = 0
    return stats
# ========================================================== #
__author__ = 'Grzegorz Knor'

app = Flask(__name__)
#app.debug = True

port = int(os.getenv("PORT", 64781))

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        image = cv2.imread('images/'+filename,1)
        figures = findSquaresAndTriangles(image);
        squares = figures['squares']
        triangles = figures['triangles']
        image = drawContours(image, squares);
        image = drawContours(image, triangles);
        i = 0
        for square in squares:        
            cv2.putText(image, str(i), (int(np.mean(square[:,0])),int(np.mean(square[:,1]))), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,0),3,cv2.CV_AA);
            i = i+1

        t = []
        i = 0
        for triangle in triangles:   
        #print triangle   
            t.append(triangleToNumber(image,triangle))
            cv2.putText(image, str(t[i]), (int((triangle[0][0]+triangle[1][0]+triangle[2][0])/3),int((triangle[0][1]+triangle[1][1]+triangle[2][1])/3)), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.CV_AA);
            i = i+1
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.savefig('images/result.jpg',dpi = 150)        
        result = trianglesToSquaresAndNumbers(triangles, squares, image)
        ar = areas(squares, 3)
        print ar
        arNo = squaresInAreas(squares, ar)
        print arNo
        d = {}
        for i in range(0,len(arNo)):
            key = str(result[arNo[i][0]])[1:-1]    
            d[key] = arNo[i][1]
        with open('images/last.json', 'w') as outfile:
            a = json.dumps(d, sort_keys=True,indent=4, separators=(',', ': '))
            json.dump(a, outfile)        
    #time.sleep( 2 )
    return render_template("index.html")
#render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

#if __name__ == "__main__":
#    app.run(port=4555, debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
