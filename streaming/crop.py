#! /usr/bin/python3

#Copyright © 2019, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
#SPDX-License-Identifier: Apache-2.0


import numpy as np
import cv2
import base64
import math

def get_all_crops_from_bbox(_image_, _nObjects_,
                            _Object0_x, _Object0_y, _Object0_width, _Object0_height, _P_Object0_,
                            _Object1_x, _Object1_y, _Object1_width, _Object1_height, _P_Object1_,
                            _Object2_x, _Object2_y, _Object2_width, _Object2_height, _P_Object2_,
                            _Object3_x, _Object3_y, _Object3_width, _Object3_height, _P_Object3_,
                            _Object4_x, _Object4_y, _Object4_width, _Object4_height, _P_Object4_,
                            _Object5_x, _Object5_y, _Object5_width, _Object5_height, _P_Object5_,
                            _Object6_x, _Object6_y, _Object6_width, _Object6_height, _P_Object6_,
                            _Object7_x, _Object7_y, _Object7_width, _Object7_height, _P_Object7_,
                            _Object8_x, _Object8_y, _Object8_width, _Object8_height, _P_Object8_,
                            _Object9_x, _Object9_y, _Object9_width, _Object9_height, _P_Object9_,
                            _Object10_x, _Object10_y, _Object10_width, _Object10_height, _P_Object10_,
                            _Object11_x, _Object11_y, _Object11_width, _Object11_height, _P_Object11_,
                            _Object12_x, _Object12_y, _Object12_width, _Object12_height, _P_Object12_,
                            _Object13_x, _Object13_y, _Object13_width, _Object13_height, _P_Object13_,
                            _Object14_x, _Object14_y, _Object14_width, _Object14_height, _P_Object14_,
                            _Object15_x, _Object15_y, _Object15_width, _Object15_height, _P_Object15_,
                            _Object16_x, _Object16_y, _Object16_width, _Object16_height, _P_Object16_,
                            _Object17_x, _Object17_y, _Object17_width, _Object17_height, _P_Object17_,
                            _Object18_x, _Object18_y, _Object18_width, _Object18_height, _P_Object18_,
                            _Object19_x, _Object19_y, _Object19_width, _Object19_height, _P_Object19_):
    "Output:  board_id, _genderage_image_, _image_, ncrops, _emotion_image_ , _RGB_image_"

    imageBufferBase64 = _image_
    numberOfObjects = _nObjects_

    # limit the max number of objects
    if numberOfObjects > 3:        
        numberOfObjects = 3 

    nparr = np.frombuffer(base64.b64decode(imageBufferBase64), dtype=np.uint8)    
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    #img_np = cv2.resize(img_np, (int(outWidth), int(outHeight)), cv2.INTER_LINEAR)

    image_h, image_w, _ = img_np.shape

    local_vars=locals()
    row={}
    for i in range(int(float(numberOfObjects))):
        row['_Object'+str(i)+'_x'] = local_vars['_Object'+str(i)+'_x']
        row['_Object'+str(i)+'_y'] = local_vars['_Object'+str(i)+'_y']
        row['_Object'+str(i)+'_width'] = local_vars['_Object'+str(i)+'_width']
        row['_Object'+str(i)+'_height'] = local_vars['_Object'+str(i)+'_height']
        row['_P_Object'+str(i)+'_'] = local_vars['_P_Object'+str(i)+'_']

    #print(row)

    genderage_list = []
    emotion_list = []
    RGB_list = []
    board_id_list = []
    img_list = []
    
    def zoomout(x_min,x_max,y_min,y_max,fudgefactor):
    
    
        crop_height = y_max - y_min 
        crop_width = x_max - x_min 
        
        x_min2 = math.floor(x_min - (crop_width*fudgefactor))
        x_max2 = math.ceil(x_max + (crop_width*fudgefactor))
        y_min2 = math.floor(y_min - (crop_height*fudgefactor))
        y_max2 = math.ceil(y_max + (crop_height*fudgefactor))
    
            
        return x_min2,x_max2,y_min2,y_max2
    
    def nptostring(image_np):
        retval, nparr_crop = cv2.imencode(".JPEG", image_np)

        img_blob_crop = np.array(nparr_crop).tostring()
            
        img_crop_base64 = base64.b64encode(img_blob_crop)
        image_str = img_crop_base64.decode('utf-8')
            
        return image_str
    
    debugfilename = "./debugcrap2.txt"
    debugfile = open(debugfilename,'a')
    #debugfile.write ("New call from the great beyond\n")
    
    
    for i in range(0, int(float(numberOfObjects))):
        #obj = row['_Object' + str(i) + '_']
                
        x = float(row['_Object' + str(i) + '_x'])
        y = float(row['_Object' + str(i) + '_y'])
        width = float(row['_Object' + str(i) + '_width'])
        height = float(row['_Object' + str(i) + '_height'])
        prob = float(row['_P_Object' + str(i) + '_'])
        probability = "(" + str(round(prob * 100, 1)) + "%)"
        
        string = "X Y width height probability "+str(x)+" "+str(y)+" "+str(width)+" "+str(height)+" "+str(prob)+"\n"
        #debugfile.write (string)
        
        rx_min = x - width  / 2
        rx_max = x + width  / 2
        ry_min = y - height / 2
        ry_max = y + height / 2
        
        x_min = int(image_w * rx_min)
        x_max = int(image_w * rx_max)
        y_min = int(image_h * ry_min)
        y_max = int(image_h * ry_max)
        
        crop_height = y_max - y_min 
        crop_width = x_max - x_min 
        
        string = "original X Xb Y Yb  width  height "+str(x_min)+" "+str(x_max)+" "+str(y_min)+" "+str(y_max)+" "+str(crop_width)+" "+str(crop_height) +"\n"
        #debugfile.write (string)
        
        
        
        # square up image and zoom out based on aspect ratio. 
        aspectratio = 1 
        if crop_width > 0 : # don't divide by 0 
            aspectratio = crop_height / crop_width
            
        deltaY = math.floor(crop_height - crop_width) / 2 
        y_max = y_min + (x_max - x_min )  # square image 
        #  move down to center image 
        y_min = y_min + deltaY
        y_max = y_max + deltaY
        # the closer the aspectratio is to 1 the less zooming out is needed 
        fudge = .08  # the smaller the number the less zooming out 
        fudge = ((aspectratio - 1 ) * fudge ) + fudge 
        
        
        string = "delta prob) aspect ratio  " +str(fudge)+" "+str(prob)+" "+str(aspectratio)+"\n"
        #debugfile.write (string)

        # CROP ORIGINAL IMAGE
        x_min4,x_max4,y_min4,y_max4 = zoomout(x_min,x_max,y_min,y_max,fudge)
        new_crop = img_np[y_min4:y_max4, x_min4:x_max4, ::-1]
        #new_crop = img_np[y_min:y_max, x_min:x_max, ::-1]
        
        x_min3,x_max3,y_min3,y_max3 = zoomout(x_min,x_max,y_min,y_max,fudge+.15)
        emotion_crop = img_np[y_min3:y_max3, x_min3:x_max3, ::-1]
        
        
        
        string = "gender X4 X4b Y4 Y4b  "+str(x_min4)+" "+str(x_max4)+" "+str(y_min4)+" "+str(y_max4)+"\n"
        #debugfile.write (string)
        string = "emotion X3 X3b Y3 Y3b  "+str(x_min3)+" "+str(x_max3)+" "+str(y_min3)+" "+str(y_max3)+"\n"
        #debugfile.write (string)
        

        if (new_crop.size > 0) and (crop_width > 50) and (prob > .853 ):
            new_crop = cv2.resize(new_crop, (224, 224), cv2.INTER_LINEAR)
            emotion_crop = cv2.resize(emotion_crop, (224, 224), cv2.INTER_LINEAR)
            RGB_crop = cv2.cvtColor(emotion_crop,cv2.COLOR_BGR2RGB)
            emotion_crop = cv2.cvtColor(emotion_crop,cv2.COLOR_BGR2GRAY)
            
            #cv2.imwrite("./pics/newcrap"+str(crop_width)+".jpg", new_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            #cv2.imwrite("./pics/emotioncrap"+str(crop_width)+".jpg", emotion_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            #cv2.imwrite("./pics/RGBcrap"+str(crop_width)+".jpg", RGB_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            board_id_list.append(i)
            
            genderage_list.append(nptostring(new_crop))
            emotion_list.append(nptostring(emotion_crop))
            RGB_list.append(nptostring(RGB_crop))

    ncrops = len(board_id_list) 
    debugfile.close()
    if ncrops > 0:
        img_list = ['MA==']*ncrops
        img_list[0] = imageBufferBase64 

    if ncrops == 0:
        return  0, None, None, None , None , None
    
    return  board_id_list, genderage_list, img_list, float(ncrops), emotion_list, RGB_list
    


