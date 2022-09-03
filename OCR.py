from genericpath import isfile
import cv2
import os
import numpy as np
import model_test

def auto_bc(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return auto_result

def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def plate_preprocess(image, orig, ROI_number):
    plate = list()                                                                              #new list to store number plate
    cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                      #finding contours in the image
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]                                               #Setting the starting countour
    (cnts, _) = sort_contours(cnts, method="left-to-right")                                     #Sort contours in descending order
    for c in cnts:                                                                              #loop through all the contours
        area = cv2.contourArea(c)                                                               #get area of the respective contour
        if area > 120 and area < 500: #and area > 100:                                          #Check area condition
            x,y,w,h = cv2.boundingRect(c)                                                       #Get coordinates of the contour
            x-=1                                                                                    
            y+=2                                                                                #Setting coordinates value to get the full cropped image
            w+=2
            h-=2
            ROI = orig[y:y+h, x:x+w]                                                            #Crop the original image with above coordinates
            # cv2.drawContours(image, [c], -1, (255,255,255), -1)
            # print(pt.image_to_string(ROI,config="--psm 6 --oem 3"))
            try: 
                ROI = cv2.erode(ROI,np.ones([1,3]),iterations = 1)                              #Erode the cropped image  
                # display(ROI,"ROI")
                ROI_gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)                                 #Convert to grayscale
                ROI_gray = cv2.resize(ROI_gray,(28, 28), interpolation = cv2.INTER_NEAREST)     #Resize the image to 28x28
                # display(ROI_gray,"ROI_gray")
                # ROI = cv2.resize(ROI,(28, 28), interpolation = cv2.INTER_NEAREST)
                # print(ROI_number,"=",area)
                x = model_test.main(ROI_gray)                                                   #Pass the ROI image to model to predict the character
                plate.append(x)                                                                 #Append the plate list with predicted value
                
            except:
                print("ROI not found")                                                          #Exception
    string = ''.join(plate)                                                                     #join all the values in list to make it a string
    return string                                                                               #return string
    
def write(img,count):
    path = 'C:\\Users\\SAMSUNG\\Desktop\\Final'
    cv2.imwrite(os.path.join(path , str(count)+'.jpg'),img)

def display(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    
def alpr_main(img):
    lower = np.array([15,150,80])                       #Setting upper value for mask
    upper = np.array([35,255,255])                      #Setting lower value for mask
    image = auto_bc(img)
    # display(image,"autobc")
    image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)   #Converting image to HSV format
    # res = thresh(image)
    mask = cv2.inRange(image_hsv, lower, upper)         #Creating mask of the hsv image using lower and upper values
    # display(mask,"mask")     
    plate = plate_preprocess(mask,image,0)              
    # img = cv2.bitwise_not(mask)
    # display(img,"invert")
    # plate_preprocess(img,image,0)
    return plate

# def main1():
#     path = "C:\\Users\\SAMSUNG\\Desktop\\Resized_Plates"
#     for files in os.listdir(path):
#         file = os.path.join(path,files)                         #join directory path with the file
#         if os.path.isfile(file):
#             print(file)
#             img = cv2.imread(file)                              #read image
#             # display(img,"img")
#             # lower = np.array([0,0,0])
#             # upper = np.array([350,55,100])
#             lower = np.array([15,150,80])                       #Setting upper value for mask
#             upper = np.array([35,255,255])                      #Setting lower value for mask
#             image = auto_bc(img)
#             # display(image,"autobc")
#             image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)   #Converting image to HSV format
#             # res = thresh(image)
#             mask = cv2.inRange(image_hsv, lower, upper)         #Creating mask of the hsv image using lower and upper values
#             # display(mask,"mask")     
#             plate = plate_preprocess(mask,image,0)              
#             # img = cv2.bitwise_not(mask)
#             # display(img,"invert")
#             # plate_preprocess(img,image,0)
#             write(img,plate)
#             cv2.destroyAllWindows()

# # def main2():
# #     file = "C:\\Users\\SAMSUNG\\Desktop\\Resized_Plates\\0.jpg"
# #     img = cv2.imread(file)
# #     display(img,"img")
# #     lower = np.array([0,0,0])
# #     upper = np.array([350,55,100])
# #     image = auto_bc(img)
# #     image = auto_bc(image)
# #     image = auto_bc(image)
# #     image = auto_bc(image)
# #     display(image,"autobc")
# #     image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# #     # res = thresh(image)
# #     mask = cv2.inRange(image_hsv, lower, upper)
# #     display(mask,"mask")     
    
# main1()