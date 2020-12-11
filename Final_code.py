import numpy as np
import cv2
import matplotlib.image as mpimage
import pickle


#     _____                                       _      _                              _ _ _               _   _             
#    / ____|                                     (_)    | |                            | (_) |             | | (_)            
#   | |     __ _ _ __ ___   ___ _ __ __ _   _ __  _  ___| |_ _   _ _ __ ___    ___ __ _| |_| |__  _ __ __ _| |_ _  ___  _ __  
#   | |    / _` | '_ ` _ \ / _ \ '__/ _` | | '_ \| |/ __| __| | | | '__/ _ \  / __/ _` | | | '_ \| '__/ _` | __| |/ _ \| '_ \ 
#   | |___| (_| | | | | | |  __/ | | (_| | | |_) | | (__| |_| |_| | | |  __/ | (_| (_| | | | |_) | | | (_| | |_| | (_) | | | |
#    \_____\__,_|_| |_| |_|\___|_|  \__,_| | .__/|_|\___|\__|\__,_|_|  \___|  \___\__,_|_|_|_.__/|_|  \__,_|\__|_|\___/|_| |_|
#                                          | |                                                                                
#                                          |_|                                                                                

global mtx, dist
calibration_pickle = pickle.load( open( "./calibration_matrix.p", "rb" ) )
mtx = calibration_pickle["mtx"]
dist = calibration_pickle["dist"]

       
#    _______ _                   _           _     _ _                                      _ _            
#   |__   __| |                 | |         | |   | (_)                                    | (_)           
#      | |  | |__  _ __ ___  ___| |__   ___ | | __| |_ _ __   __ _   _ __  _   _ _ __   ___| |_ _ __   ___ 
#      | |  | '_ \| '__/ _ \/ __| '_ \ / _ \| |/ _` | | '_ \ / _` | | '_ \| | | | '_ \ / _ \ | | '_ \ / _ \
#      | |  | | | | | |  __/\__ \ | | | (_) | | (_| | | | | | (_| | | |_) | |_| | |_) |  __/ | | | | |  __/
#      |_|  |_| |_|_|  \___||___/_| |_|\___/|_|\__,_|_|_| |_|\__, | | .__/ \__, | .__/ \___|_|_|_| |_|\___|
#                                                             __/ | | |     __/ | |                        
#                                                            |___/  |_|    |___/|_|     

def sobel(image, coordinate='unspecified', thresh=(20, 255)):
     
    #laplacian = cv2.Laplacian(imgae,cv2.CV_64F)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if coordinate == 'x':
        sobel = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    if coordinate == 'y':
        sobel = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        
    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

def color_thresh(image, HSV_thresh=(0,255)):

    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = HSV[:,:,1]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel>=HSV_thresh[0]) & (v_channel<=HSV_thresh[1])] = 1   
    
    c_binary = np.zeros_like(v_channel)
    c_binary[(v_binary==1)] = 1
    
    return c_binary
    
def thresh_pipeline(image, gradx_thresh, grady_thresh, HSV_thresh):
    
    gradx = sobel(image, coordinate='x', thresh=gradx_thresh)
    grady = sobel(image, coordinate='y', thresh=grady_thresh)
    
    c_binary = color_thresh(image, HSV_thresh)
    
    thresh_binary = np.zeros_like(image[:,:,0])
    thresh_binary[(gradx==1) & (grady==1) | (c_binary==1) ] = 255
    
    return thresh_binary

#    _____                              _   _             _                        __                     
#   |  __ \                            | | (_)           | |                      / _|                    
#   | |__) |__ _ __ ___ _ __   ___  ___| |_ ___   _____  | |_ _ __ __ _ _ __  ___| |_ ___  _ __ _ __ ___  
#   |  ___/ _ \ '__/ __| '_ \ / _ \/ __| __| \ \ / / _ \ | __| '__/ _` | '_ \/ __|  _/ _ \| '__| '_ ` _ \ 
#   | |  |  __/ |  \__ \ |_) |  __/ (__| |_| |\ V /  __/ | |_| | | (_| | | | \__ \ || (_) | |  | | | | | |
#   |_|   \___|_|  |___/ .__/ \___|\___|\__|_| \_/ \___|  \__|_|  \__,_|_| |_|___/_| \___/|_|  |_| |_| |_|
#                      | |                                                                                
#                      |_|                                                                                

### This part of the code is for both lane lines - right and left

line_l_detected = False
line_r_detected = False

line_l_recent_xfitted = []
line_r_recent_xfitted = []
   
line_l_best_fit = None
line_r_best_fit = None 
 
line_l_diffs = np.array([0,0,0], dtype='float')
line_r_diffs = np.array([0,0,0], dtype='float') 
 
def perspective():
    global src, dst, M, Minv
    
    image_width = 1280
    image_height = 720

    bottom_width = 0.75 
    mid_width = 0.17 
    height_pct = 0.66 
    bottom_trim = 0.935
    
    src = np.float32([
        [image_width*(0.5-mid_width/2), image_height*height_pct],
        [image_width*(0.5+mid_width/2), image_height*height_pct],
        [image_width*(0.5+bottom_width/2), image_height*bottom_trim],
        [image_width*(0.5-bottom_width/2), image_height*bottom_trim]
    ])
    
    offset = image_width*0.2
    
    dst = np.float32([
        [offset, 0],
        [image_width-offset, 0],
        [image_width-offset, image_height],
        [offset, image_height]
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src) 

def transform_wrapper(image, src, dst):
    
    image_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

#    _                            _      _            _   _             
#   | |                          | |    | |          | | (_)            
#   | |     __ _ _ __   ___    __| | ___| |_ ___  ___| |_ _  ___  _ __  
#   | |    / _` | '_ \ / _ \  / _` |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#   | |___| (_| | | | |  __/ | (_| |  __/ ||  __/ (__| |_| | (_) | | | |
#   |______\__,_|_| |_|\___|  \__,_|\___|\__\___|\___|\__|_|\___/|_| |_|
#                                                                       
#                                                                       

def lane_boundary(image):
    binary_warped = image.astype('uint8')
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_image = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]/2)
    left_x_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

   
    search_windows = 20
    
    window_height = np.int(binary_warped.shape[0]/search_windows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_x_current = left_x_base
    rightx_current = rightx_base
    
    margin = 70
    minpix = 75
    
    # Left and right lane pixels ->
    left_lane_pixels = []
    right_lane_pixels = []

    for window in range(search_windows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_image,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_image,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_pixels.append(good_left_inds)
        right_lane_pixels.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_pixels = np.concatenate(left_lane_pixels)
    right_lane_pixels = np.concatenate(right_lane_pixels)

    # Left and right lane pixel positions
    left_x = nonzerox[left_lane_pixels]
    left_y = nonzeroy[left_lane_pixels] 
    right_x = nonzerox[right_lane_pixels]
    right_y = nonzeroy[right_lane_pixels] 

    
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    out_image[nonzeroy[left_lane_pixels], nonzerox[left_lane_pixels]] = [20, 255,255]
    out_image[nonzeroy[right_lane_pixels], nonzerox[right_lane_pixels]] = [255, 0,0]
    
    return left_fit, right_fit, out_image

def lane_following(image, left_fit, right_fit):
  
    binary_warped = image.astype('uint8')

    # output image
    out_image = np.dstack((binary_warped, binary_warped, binary_warped))
    window_image = np.zeros_like(out_image)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    left_lane_pixels = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_pixels = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    left_x = nonzerox[left_lane_pixels]
    left_y = nonzeroy[left_lane_pixels] 
    right_x= nonzerox[right_lane_pixels]
    right_y = nonzeroy[right_lane_pixels]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_image[nonzeroy[left_lane_pixels], nonzerox[left_lane_pixels]] = [255, 0, 0]
    out_image[nonzeroy[right_lane_pixels], nonzerox[right_lane_pixels]] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_image, 1, window_image, 0.3, 0)
    
    return left_fit, right_fit, result

def lane_curvature(ploty, left_fitx, right_fitx):
    
    y_eval = np.max(ploty)
    
    global ym_per_pix, xm_per_pix
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/505/1.2054/0.97 # meters per pixel in x dimension
 
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    lane_center = (left_fitx[-1]+right_fitx[-1])/2
        
    center_diff = (640-lane_center)*xm_per_pix
    
    return left_curverad, right_curverad, center_diff

def lane_canvas(image_undist, binary_warped, Minv, ploty, left_fitx, right_fitx):
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    ## This is the lane line color-coding part->
    
    cv2.fillPoly(color_warp, np.int_([pts]), (50,200, 50))
    
    middle_x = (left_fitx + right_fitx)/2
    middle_pts = np.transpose(np.vstack((middle_x, ploty))).astype(np.int32)
    
    cv2.polylines(color_warp, np.int32([middle_pts]), False, (0, 0, 255), thickness=30)
    cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), thickness=30)
    cv2.polylines(color_warp, np.int32([pts_right]), False, (255, 0, 0), thickness=30)
    
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_undist.shape[1], image_undist.shape[0])) 
    result = cv2.addWeighted(image_undist, 1, newwarp, 0.5, 5)
    
    return result

def lane_quality(left_fitx, right_fitx):
    
    lane_width = (right_fitx - left_fitx)
    lane_width_mean = np.mean(lane_width)*xm_per_pix
    lane_width_var = np.var(lane_width)
    
    return lane_width_mean, lane_width_var

def lane_finding(image_orig):
    
    # 2. Apply a distortion correction to raw images.
    image_undist = cv2.undistort(image_orig, mtx, dist, None, mtx)
    
    # 3. Use color transforms, gradients, etc., to create a thresholded binary image.
    image_thresh = thresh_pipeline(image_undist, gradx_thresh=(25,255), grady_thresh=(10,255), HSV_thresh=(175, 255))
    
    # 4. Apply a perspective transform to rectify binary image ("birds-eye view").
    # src, dst, M, Minv are global variables

    image_birdeye = transform_wrapper(image_thresh, src, dst)
    image_birdeye_color = transform_wrapper(image_undist, src, dst)
    
    # 5. Detect lane pixels and fit to find the lane boundary. 
    line_l_detected = False
    line_r_detected = False
    
    if (not line_l_detected) or (not line_r_detected):
        # Run a sliding windows search
        left_fit, right_fit, image_search = lane_boundary(image_birdeye)
        
        cv2.putText(image_search, 'Following', 
                    (550, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
  
    else:
        
        left_fit, right_fit, image_search = lane_following(image_birdeye, 
                                                        line_l.recent_xfitted[-1][0], 
                                                        line_r.recent_xfitted[-1][0]) 
        cv2.putText(image_search, 'Not following', 
                    (550, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
    
    line_l_recent_xfitted.append([left_fit])
    line_r_recent_xfitted.append([right_fit])  

    if len(line_l_recent_xfitted)>1:
        
        line_l_best_fit = np.mean(np.array(line_l_recent_xfitted[-20:-1]),
                                axis=0) 
        line_r_best_fit = np.mean(np.array(line_r_recent_xfitted[-20:-1]),
                                axis=0)    
    else:
        line_l_best_fit = line_l_recent_xfitted[-1][0]
        line_r_best_fit = line_r_recent_xfitted[-1][0]  
            
    ploty = np.linspace(0, image_birdeye.shape[0]-1, image_birdeye.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    lane_curvature(ploty, left_fitx, right_fitx)
    lane_width_mean, lane_width_var = lane_quality(left_fitx, right_fitx)
    
    line_l_diffs = left_fit - line_l_best_fit
    
    line_r_diffs = right_fit - line_r_best_fit
    
    lane_continue = np.sum(line_l_diffs**2)+np.sum(line_r_diffs)
    
    if (not 3<lane_width_mean<5) or (lane_width_var>500) or (lane_continue>6000):
        
        del line_l_recent_xfitted[-1]
        
        del line_r_recent_xfitted[-1]
        
        left_fit, right_fit = line_l_best_fit[0], line_r_best_fit[0]
                
    else:
        line_l_detected = True
        line_r_detected = True

        line_l_best_fit = np.mean(np.array(line_l_recent_xfitted[-20:]),
                                axis=0) 
        line_r_best_fit = np.mean(np.array(line_r_recent_xfitted[-20:]),
                                axis=0)
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    lane_curvature(ploty, left_fitx, right_fitx)
    lane_width_mean, lane_width_var = lane_quality(left_fitx, right_fitx)
        
    result = lane_canvas(image_undist, image_birdeye, Minv, ploty, left_fitx, right_fitx)
    
    cv2.putText(result, "Threshold output" , 
                (115, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result, "Birds-eye output" , 
                (515, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result, "Lane searching" , 
                (920, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    
    image_canvas = np.ones([720,1280,3], dtype=np.uint8) ## image_canvas size
    image_canvas[0:720, 0:1280, :] = result
    
    
    ## Plot 1 - threshold image
    image_debug = image_thresh
    color_debug = np.dstack(( image_debug, image_debug, image_debug ))
    cv2.polylines(color_debug, np.int32([src]), True, (0, 255, 0), thickness=4)
    plot1 = cv2.resize(color_debug, (300, 200))
    
    ## Plot 2  
    plot2 = image_search
    left_pts = np.transpose(np.vstack((left_fitx, ploty))).astype(np.int32)    
    right_pts = np.transpose(np.vstack((right_fitx, ploty))).astype(np.int32)
    cv2.polylines(plot2, np.int32([left_pts]), False, (255, 255, 0), thickness=5)
    cv2.polylines(plot2, np.int32([right_pts]), False, (255, 255, 0), thickness=5)
    plot2= cv2.resize(image_search, (300, 200))
    
    ## Plot 3: birds eye view
    
    cv2.polylines(image_birdeye_color, np.int32([dst]), True, (0, 255, 0), thickness=4)
    plot3 = cv2.resize(image_birdeye_color, (300, 200))
    
    image_canvas[50:250, 100:400] = plot1
    image_canvas[50:250, 900:1200] = plot2
    image_canvas[50:250, 500:800] = plot3
    
    return image_canvas

perspective()

#     _____                 _      _            _   _             
#    / ____|               | |    | |          | | (_)            
#   | |     __ _ _ __    __| | ___| |_ ___  ___| |_ _  ___  _ __  
#   | |    / _` | '__|  / _` |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#   | |___| (_| | |    | (_| |  __/ ||  __/ (__| |_| | (_) | | | |
#    \_____\__,_|_|     \__,_|\___|\__\___|\___|\__|_|\___/|_| |_|
#                                                                 
#                                                                 

car=cv2.CascadeClassifier("cars.xml")

def ProcessedFrame(frame):
    grayed=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average=np.mean(grayed)
    ret,thresh = cv2.threshold(grayed,average+30,255,cv2.THRESH_BINARY)
    return thresh

def DetectedVehicle(frame,classifier):
    grayed=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars=car.detectMultiScale(grayed,1.1,1)
    thresh=ProcessedFrame(frame)
    for (x,y,w,h) in cars:
        print(x)
        
        if x > 1100:
            x = 1000
            
        if thresh[y][x]== 255 and thresh[y+h][x+w]==255 and list(thresh[y][x:x+w])!=[255]*len(list(thresh[y][x:x+w])) and h>25:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, 'Vehicle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        if  len(thresh[y:y+h][x:x+w])==0:
            continue
        unique, counts=np.unique(thresh[y:y+h][x:x+w], return_counts=True)
        if len(counts)==1 and unique==0:
            continue
        elif len(counts)==1:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, 'Vehicle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            continue
        if counts[1]>counts[0] and w*h<11000:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.putText(frame, 'Vehicle', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
        else:
            break


#   __      ___     _                                      _        _   _             
#   \ \    / (_)   | |                                    | |      | | (_)            
#    \ \  / / _  __| | ___  ___     __ _ _ __  _ __   ___ | |_ __ _| |_ _  ___  _ __  
#     \ \/ / | |/ _` |/ _ \/ _ \   / _` | '_ \| '_ \ / _ \| __/ _` | __| |/ _ \| '_ \ 
#      \  /  | | (_| |  __/ (_) | | (_| | | | | | | | (_) | || (_| | |_| | (_) | | | |
#       \/   |_|\__,_|\___|\___/   \__,_|_| |_|_| |_|\___/ \__\__,_|\__|_|\___/|_| |_|
#                                                                                     
#                                                                                     


# Video to annotate
cap = cv2.VideoCapture('project_video.mp4')
#cap = cv2.VideoCapture('project.mp4')
#cap = cv2.VideoCapture('test1.mp4')
 
if (cap.isOpened() == False): 
    print("Error opening video stream or file")

# Read until video is completed
print("Video started")
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        #thresh=ProcessedFrame(frame) 
        DetectedVehicle(frame,car)
        frame2 = lane_finding(frame)
        cv2.imshow('Frame', frame2)
        #cv2.imshow("cardet",frame)
        #cv2.imshow("Thershold image",thresh)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print("Video ended")
cap.release()
cv2.destroyAllWindows()


