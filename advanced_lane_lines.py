# Import necessary libraries
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import argparse
import glob
import os

# Create arguments to run on correct video and save new video with lane lines to a directory
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add lane lines to videos')
    parser.add_argument(
        'which_video',
        type=str,
        nargs='?',
        default='project',
        help='type project for project video or challenge for challenge video.'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',
        default='output_videos/',
        help='Path to output video folder. This is where the video from the run will be saved.'
    )
    args = parser.parse_args()
    
# Create function to calibrate camera, by finding corners on a number of chessboard images in 'camera_cal' directory.
# Using those images where 9x6 inner corners were detected, create 'undistort' function. Since 'mtx' 
# and 'dist' variables never change as long as you're using the same camera, create constants.
def calibrate_cam(direct, nx=9, ny=6, plot=False):
    fns = glob.glob(direct)
    objpts, imgpts = [], []
    if plot == True:
        plt.figure(figsize=(25,25))
    for i in range(len(fns)):
        img = mpimg.imread(fns[i])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        objpt = np.zeros((nx*ny, 3), np.float32)
        objpt[:, :2] = np.mgrid[:nx,:ny].T.reshape(-1,2)
        ret, corners = cv2.findChessboardCorners(gray, (ny,nx), None)
        if ret == True:
            imgpts.append(corners)
            objpts.append(objpt)
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        if plot == True:
            plt.subplot(5,4,(i+1))
            plt.imshow(img)
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs

check_direct = 'camera_cal/calibration*.jpg'
mtx, dist, rvecs, tvecs = calibrate_cam(check_direct)

def undistort(img, m=mtx, d=dist):
    return cv2.undistort(img, m, d, None, mtx)

# Transform each image to bird's-eye view. The vertices are found manually using a stretch of straight road in the video.
# The vertices change depending on the 'which_video' argument  because a slightly longer stretch of lane, 15 pixels, looks
# and works better in the 'project' video.
def birds_eye(img, inv=False):
    if args.which_video == 'project':
        src = np.float32([[555 , 460], [740 , 460], [0, 720], [1280, 720]])
    else:
        src = np.float32([[555 , 475], [740 , 475], [0, 720], [1280, 720]])
    dst = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
    if inv:
        M = cv2.getPerspectiveTransform(dst, src)
        out_img = img
    else:
        M = cv2.getPerspectiveTransform(src, dst)
        out_img = np.zeros((img.shape[0], img.shape[1], 3))
        out_img[:,:,0] = img
    shp = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(out_img, M, shp, flags=cv2.INTER_LINEAR)
    return warped

# Create function to create a binary image by thresholding red and green channels in RGB image. The thresholds were 
# chosen to focus on yellow-to-gray stretches and works pretty well with shadows and across different road colors.
def yellow_select(img, thresh=(195,195,255)):
    z = np.zeros_like(img)
    z[(img[:,:,0] >= thresh[0]) & (img[:,:,1] >= thresh[1]) & (img[:,:,2] <= thresh[2])] = 1
    out = z[:,:,0] + z[:,:,1] + z[:,:,2]
    out[out > 1] = 1
    return out

# Detect edges by converting an image to HLS colorspace and thresholding the saturation channel. Create binary image.
def hls_select(img, thresh=(75, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

# Detect edges using a sobel filter. Create binary image. This is called twice below for both x and y orientation.
def abs_sobel_thresh(img, sobel_kernel=15, orient='x', thresh_min=10, thresh_max=50):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled = np.uint(sobel*255/np.max(sobel))
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled > thresh_min) & (scaled < thresh_max)] = 1
    return binary_output

# Combine above functions to create a binary image. Any detection from 'yellow_select' goes into final image. 
# Additional white pixels are included, but only those that were dected in all three other binary images.
def combined_grads(img):
    sobel_x = abs_sobel_thresh(img, sobel_kernel=15, orient='x', thresh_min=10, thresh_max=50)
    sobel_y = abs_sobel_thresh(img, sobel_kernel=15, orient='y', thresh_min=10, thresh_max=50)
    yellow = yellow_select(img, thresh=(175,175,255))
    hls = hls_select(img, thresh=(75, 255))
    combined = np.zeros_like(sobel_x)
    combined[(yellow == 1) | ((sobel_x==1) & (sobel_y==1) & (hls==1))] = 1
    return combined

# Create lists that will allow us to keep the polynomial coefficients of past calculations to use a moving average
# to fit the lane curves. The ys and xs of each lane are kept to find the radius of the curve even on the occasional
# frame when a line on one side isn't detected. The radii list is used to print a moving average of the radius of the 
# curve over the video because that number jumps around too much to be useful without smoothing.
l_a, l_b, l_c, r_a, r_b, r_c = [], [], [], [], [], []
left_ys, left_xs, right_ys, right_xs = [], [], [], []
radii = []

# Combine above functions to create a binary image from bird's-eye view. 
def draw_lane_lines(orig_img):
    mv_avg = 15
    undistort_img = undistort(orig_img)
    blur = cv2.GaussianBlur(undistort_img, (5, 5), 0)
    grads_img = combined_grads(blur)
    binary_warped = birds_eye(grads_img)[:,:,0]
    
    # The base points of the curves of each frame are found by the average position of white pixels in the bottom 
    # half of the warped binary image.
    histogram = np.sum(binary_warped[(binary_warped.shape[0]//2):, :], axis=0)
    out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3))
    out_img[:,:,:] = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]/2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Scroll through windows 80 frames high and 200 frames wide for both lanes, finding x_coordinates where white 
    # appears in the window, resetting the center of the window based on the average white x's of the previous window,
    # but only if at least 50 white pixels were found in that window.
    n_windows = 9
    window_height = np.int(binary_warped.shape[0]/n_windows)
    notzero = binary_warped.nonzero()
    notzero_y = np.array(notzero[0])
    notzero_x = np.array(notzero[1])

    left_x_current = left_x_base
    right_x_current = right_x_base
    margin = 100
    minpix = 50
    left_lane_inds, right_lane_inds = [], []
    for window in range(n_windows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin
        good_left_inds = ((notzero_y >= win_y_low) & (notzero_y < win_y_high) & 
                          (notzero_x >= win_xleft_low) & (notzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((notzero_y >= win_y_low) & (notzero_y < win_y_high) & 
                          (notzero_x >= win_xright_low) & (notzero_x < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(notzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = np.int(np.mean(notzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    left_x = notzero_x[left_lane_inds]
    left_y = notzero_y[left_lane_inds]
    right_x = notzero_x[right_lane_inds]
    right_y = notzero_y[right_lane_inds]

    # Fit a line only if 2500 white pixels were found in window search, so only confident lines are
    # collected and used during the moving average calculation. Only collect,  x's and y's for frames
    # where line predictions are confident, helping to smooth the radius of the curve.
    plot_y = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0], dtype=np.int)
    if (len(left_lane_inds) > 2500):
        left_fit = np.polyfit(left_y, left_x, 2)
        left_ys.append(left_y)
        left_xs.append(left_x)
        l_a.append(left_fit[0])
        l_b.append(left_fit[1])
        l_c.append(left_fit[2])
    if (len(right_lane_inds) > 2500):
        right_fit = np.polyfit(right_y, right_x, 2)
        right_ys.append(right_y)
        right_xs.append(right_x)
        r_a.append(right_fit[0])
        r_b.append(right_fit[1])
        r_c.append(right_fit[2])
    
    # Fit to lane lines of current frame with a moving average of the previous 15 confindent lines.
    left_fit_x = list(np.array(np.nanmean(l_a[-mv_avg:])*(plot_y**2) + 
                               np.nanmean(l_b[-mv_avg:])*plot_y + 
                               np.nanmean(l_c[-mv_avg:]), dtype=np.int))
        
    right_fit_x = list(np.array(np.nanmean(r_a[-mv_avg:])*(plot_y**2) + 
                                np.nanmean(r_b[-mv_avg:])*plot_y + 
                                np.nanmean(r_c[-mv_avg:]), dtype=np.int))
    
    # Approximate meters per pixel in the y direction, assuming the length of the highlighted lane is 30 meters.
    # Calculate in x direction, assuming lane width is 3.7 meters. Find radius of curve based on last set of 
    # confidently detected x and y values. Caclulate moving average of radius with the past 15 frames to print over 
    # the video.
    ym_per_pix = 30/720.
    xm_per_pix = 3.7/(right_fit_x[0] - left_fit_x[0])
    y_eval = binary_warped.shape[0]    
    left_x_wrld = left_xs[-1]
    left_y_wrld = left_ys[-1]
    right_x_wrld = right_xs[-1]
    right_y_wrld = right_ys[-1]
    left_fit_wrld = np.polyfit(left_y_wrld*ym_per_pix, left_x_wrld*xm_per_pix, 2)
    right_fit_wrld = np.polyfit(right_y_wrld*ym_per_pix, right_x_wrld*xm_per_pix, 2)
        
    left_curverad = ((1 + (2*left_fit_wrld[0]*y_eval*ym_per_pix + 
                           left_fit_wrld[1])**2)**1.5) / np.abs(2*left_fit_wrld[0])
        
    right_curverad = ((1 + (2*right_fit_wrld[0]*y_eval*ym_per_pix + 
                            right_fit_wrld[1])**2)**1.5) / np.abs(2*right_fit_wrld[0])
    radius = np.mean([left_curverad, right_curverad])
    radii.append(radius)
    current_r =np.mean(radii[-mv_avg:])
    
    # Create overlay image so that the lane is highlighted in green. I use the for loop to make the lane boundaries red.
    # I was worried, so I double-checked and this does not slow down the algorithm compared to using cv2.polylines.
    overlay = np.zeros_like(out_img, dtype=np.uint8)
    for i in range(len(overlay)):
        overlay[plot_y[i],left_fit_x[i]:left_fit_x[i]+25, 0] = 255
        overlay[plot_y[i],right_fit_x[i]-25:right_fit_x[i], 0] = 255
        overlay[plot_y[i],left_fit_x[i]+25:right_fit_x[i]-25, 1] = 255
    
    # Convert to original perspective and add overlay to original.
    overlay_unbirded = birds_eye(overlay, inv=True)
    orig_with_lanes = cv2.addWeighted(undistort_img, 1.0, overlay_unbirded, 0.33, 0)
    
    # Print out the radius of the curve, the car's distance from the center of the lane, and a line indicating the 
    # car's center over the frame.
    car_cntr = orig_with_lanes.shape[1]//2
    dst_frm_cntr = ((car_cntr - left_fit_x[-1]) - (right_fit_x[-1] - car_cntr)) * 0.5 * xm_per_pix
    cv2.putText(orig_with_lanes,'Dist from lane center: {:.2f}m'.format(np.abs(dst_frm_cntr)), (400, 625), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1,(0,0,255),4,cv2.LINE_AA)
    cv2.putText(orig_with_lanes,'Radius of curve: {:.2f}m'.format(current_r), (225, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),7,cv2.LINE_AA)
    cv2.putText(orig_with_lanes,'car center', (575, orig_with_lanes.shape[0]-58), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    cv2.line(orig_with_lanes,(640, orig_with_lanes.shape[0]-50), (640, orig_with_lanes.shape[0]), [0,0,255], 4)
    return orig_with_lanes

# Create directory to save new video and process each frame of the project or challenge video with 'draw_lane_lines'.
if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

project_video = args.output_dir+args.which_video+'_video.mp4'
clip1 = VideoFileClip(args.which_video+'_video.mp4')
clip_lines = clip1.fl_image(draw_lane_lines)
clip_lines.write_videofile(project_video, audio=False)