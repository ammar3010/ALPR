from cv2 import COLOR_BGR2GRAY
import pytesseract as pt
import cv2
import numpy as np
import os
from OCR import alpr_main

def display(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def find_contours(image,mask):
    orig = image.copy()
    crop_img = None                                                                             #function to find contours
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img1 = image.copy()
    cv2.drawContours(img1,contours,-1,(0,255,0),2)
    contours=sorted(contours, key = cv2.contourArea, reverse = True)[:10]                       #sorting the contours according to their area, and then take the top 30 contours
    Number_Plate_Contour = 0
    for contour in contours:                                                                    #looping through all the counters
        perimeter = cv2.arcLength(contour, True)                      
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)   
        if (len(approx)==4):                                                                    #targeting only those contours which have four points
            Number_Plate_Contour = approx
            x, y, w, h = cv2.boundingRect(Number_Plate_Contour)                                 #taking contour coordinates by bounding with rectangle
            crop_img = orig[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)                        #drawing rectangle on the original image wrt contour coordinates        
            return image,crop_img,x,y                                                           #returning the mask and rectangle drawn image
        else:
            bfilter = cv2.bilateralFilter(image,15,35,35)                                       #to reduce noise
            bfilter_hsv = cv2.cvtColor(bfilter,cv2.COLOR_BGR2HSV)
            mask_bfilter = cv2.inRange(bfilter_hsv, lower, upper)   
            contours, hierarchy = cv2.findContours(mask_bfilter,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            img1 = image.copy()
            cv2.drawContours(img1,contours,-1,(0,255, 0),2)                                     
            for contour in contours:
                if cv2.contourArea(contour) > 600:                                              #targeting only those contours whose area is > 600
                    x, y, w, h = cv2.boundingRect(contour)                                      #taking contour coordinates by bounding with rectangle
                    x-=2
                    y+=2
                    crop_img = orig[y:y+h, x:x+w]
                    rec_image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3) 
                    return rec_image,crop_img,x,y            
                
def write(img,count):
    path = 'C:\\Users\\SAMSUNG\\Desktop\\Plate2'
    cv2.imwrite(os.path.join(path , str(count)+'.jpg'),img)
    count-=1
    return count

if __name__=="__main__":
    crop = None
    count = 0
    path = "C:\\Users\\SAMSUNG\\Desktop\\Cars"
    for file in os.listdir(path):
        f = os.path.join(path,file)
        if os.path.isfile(f):
            print(f)
            image = cv2.imread(f)                                                               #reading image using opencv
            orig = image.copy()
            # image = rotate(image)
            lower = np.array([15, 150, 80])                                                     #setting lower bound of yellow color using H(HUE)S(SATURATION)V(VALUE)
            upper = np.array([35, 255, 255])                                                    #setting upped bound of yellow color
            image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)                                   #converting image to hsv color format
            mask = cv2.inRange(image_hsv, lower, upper)                                         #masking the HSV image according to the color boundaries so that the desired area is
                                                                                                #highlited
            try:
                image,crop,x,y = find_contours(image,mask)
                # crop = cv2.resize(crop,(154, 72), interpolation = cv2.INTER_NEAREST) 
                # count = write(crop,count)
                plate = alpr_main(crop)
                cv2.putText(img = orig, text = plate, org=(x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)
                display("Final", orig)
            except:
                print("Blank Screen Error")
                 
            # if crop is None:
            #     print("No ROI found")
            # else:
                # x = 4
                # count = write(crop,count)
                # gray = cv2.cvtColor(crop,COLOR_BGR2GRAY)
                # F_img = plate_preprocess(gray,image,x,y)
                # if F_img is not None:
                #     count = write(image,count)
                #     # write(crop,count)
                # else:
                #     print("Number plate no EXTRACTED")
        else:
            print("Error while loading file")