
# pylint: disable=import-error
import cv2
import keyboard

import numpy as np
from matplotlib import pyplot as plt

import os



def denoise_frame(frame):
    """ Function for denoising 
    image with blurring kernel """   
    
     # Create 2x2 kernel
    kernel = np.ones((2, 2), np.float32) / 4  
    
    # Applying filter on frame
    denoised_frame = cv2.filter2D(frame, -1, kernel)   
    
    return denoised_frame   # Return denoised frame


def region_of_interest(frame):
    """ Function for drawing region 
    of interest on original frame """
    
    # Get size of the original frame
    height = frame.shape[0]
    width = frame.shape[1]    
    # Create mask with 0s filled
    mask = np.zeros_like(frame)
    
    # Draw polygon

    c3=(width-50,int(height*0.65))     # top right
    c1=(50,height-10)                            # bottom left
    c2=(width-50,height-10)                        # bottom right
    c4=(50,int(height*0.65))     # top left
    region_of_interest_vertices = [c1, c2, c3,c4]

    polygon=np.array([region_of_interest_vertices], np.int32)
    # [(0, height), (width, height), (width, 0), (width/4,height/2)]

    # Fill polygon with value 255 (white color)
    cv2.fillPoly(mask, polygon, 255)
    
    # AND bitwise the mask and the original frame
    roi = cv2.bitwise_and(frame, mask)
    
    return roi    # Return region of interest


def detect_edges(frame):
    """ Function for detecting edges 
    on frame with Canny Edge Detection """ 
    
    # Convert frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    # Apply Canny edge detection function with thresh ratio 1:3
    canny_edges = cv2.Canny(frame, 50, 150)  
    # canny_edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    return canny_edges  # Return edged frame



def warp_perspective(frame):
    """ Function for warping the frame 
    to process it on skyview angle """
    
    # Get image size
    height = frame.shape[0]
    width = frame.shape[1]        
    # Offset for frame ratio saving
    offset = 30   

    # c3=(int(width*(5/6)),int(height*(3/6))) # Top-left
    # c1=(0,height-50)                        # Bottom-left point
    # c2=(width,height-50)                    # Bottom-right point
    # c4=(int(width*(1/6)),int(height*(3/6))) # Top-right point

    c3=(int(520),int(530)) # Top-left
    c1=(50,630)                        # Bottom-left point
    c2=(1190,630)                    # Bottom-right point
    c4=(int(730),int(530)) # Top-right point
    region_of_interest_vertices = [c1, c2, c3,c4]

    polygon=np.array([region_of_interest_vertices], np.int32)
    # Perspective points to be warped
    # source_points = np.float32([[int(width*0.46), int(height*0.72)], # Top-left point
    #                   [int(width*0.58), int(height*0.72)],           # Top-right point
    #                   [int(width*0.30), height],                     # Bottom-left point
    #                   [int(width*0.82), height]])                    # Bottom-right point
    

    # source_points = np.float32([[int(width*(5/6)),int(height*(3/6))], # Top-left point
    #                   [int(width*(1/6)),int(height*(3/6))],           # Top-right point
    #                   [0,height-50],                     # Bottom-left point
    #                   [width,height-50]])   

    # frame = cv2.circle(frame, (0,430), 5, (0,0,255), 10) # Bottom-left point
    # frame = cv2.circle(frame, (640,430), 5, (0,0,255), 10) # Bottom-right point
    # frame = cv2.circle(frame, (40,220), 5, (0,0,255), 10)   # Top-left point
    # frame = cv2.circle(frame, (600,220), 5, (0,0,255), 10)  # Top-right point

    source_points = np.float32(region_of_interest_vertices)

    # source_points = np.float32([[100,300], # Top-left point
    #                             [520,300],  # Top-right point
    #                             [0,430],    # Bottom-left point
    #                             [640,430]   # Bottom-right point
    #                             ])     

    # Window to be shown
    destination_points = np.float32([[offset, 0],                    # Top-left point
                      [width-2*offset, 0],                           # Top-right point
                      [offset, height],                              # Bottom-left point
                      [width-2*offset, height]])                     # Bottom-right point
    
    # Matrix to warp the image for skyview window
    matrix = cv2.getPerspectiveTransform(source_points, destination_points) 
    
    # Final warping perspective 
    skyview = cv2.warpPerspective(frame, matrix, (width, height))    

    return skyview    # Return skyview frame





def optimize_lines(frame, lines):
    """ Function for line optimization and 
    outputing one solid line on the road """
    
    height = frame.shape[0]
    width = frame.shape[1]     

    if lines is not None:   # If there no lines we take line in memory
        # Initializing variables for line distinguishing
        lane_lines = [] # For both lines
        left_fit = []   # For left line
        right_fit = []  # For right line
        
        for line in lines:  # Access each line in lines scope
            x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates

            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Take parameters from points gained
            slope = parameters[0]       # First parameter in the list parameters is slope
            intercept = parameters[1]   # Second is intercept
            
            if slope < 0:   # Here we check the slope of the lines 
                left_fit.append((slope, intercept))
            else:   
                right_fit.append((slope, intercept))

        if len(left_fit) > 0:       # Here we ckeck whether fit for the left line is valid
            left_fit_average = np.average(left_fit, axis=0)     # Averaging fits for the left line
            lane_lines.append(map_coordinates(frame, left_fit_average)) # Add result of mapped points to the list lane_lines
            
        if len(right_fit) > 0:       # Here we ckeck whether fit for the right line is valid
            right_fit_average = np.average(right_fit, axis=0)   # Averaging fits for the right line
            lane_lines.append(map_coordinates(frame, right_fit_average))    # Add result of mapped points to the list lane_lines
        
    return lane_lines       # Return actual detected and optimized line 



def display_lines(frame, lines):
    """ Function for displaying 
    lines on the original frame """
    
    # Create array with zeros using the same dimension as frame
    mask = np.zeros_like(frame)   
    
    if lines is not None:                   # Check if there is a existing line
        for line in lines:                  # Iterate through lines list
            for x1, y1, x2, y2 in line:     # Unpack line by coordinates
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 7)    # Draw the line on the created mask
    
    # Merge mask with original frame
    frame = cv2.addWeighted(frame, 1, mask, 1, 1)    
    
    return frame    # Return frame with displayed lines



def histogram(frame):
    """ Function for histogram 
    projection to find leftx and rightx bases """
    
    # Build histogram
    histogram = np.sum(frame, axis=0)   
    
    # Find mid point on histogram
    midpoint = np.int(histogram.shape[0]/2)    
    
    # Compute the left max pixels
    left_x_base = np.argmax(histogram[:midpoint])   
    
    # Compute the right max pixels
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint  
    
    # print(left_x_base)
    # print(right_x_base)

    return left_x_base, right_x_base    # Return left_x and right_x bases


def map_coordinates(frame, parameters):
    """ Function for mapping given 
    parameters for line construction """
    
    height = frame.shape[0]
    width = frame.shape[1]  
    slope, intercept = parameters   # Unpack slope and intercept from the given parameters
    
    if slope == 0:      # Check whether the slope is 0
        slope = 0.1     # handle it for reducing Divisiob by Zero error
    
    y1 = height             # Point bottom of the frame
    y2 = int(height*0.72)  # Make point from middle of the frame down  
    x1 = int((y1 - intercept) / slope)  # Calculate x1 by the formula (y-intercept)/slope
    x2 = int((y2 - intercept) / slope)  # Calculate x2 by the formula (y-intercept)/slope
    
    return [[x1, y1, x2, y2]]   # Return point as array



def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img



def run_car():
    """
    Function to c
"""


    folder = 'data'
    print(type(folder))
    files = os.listdir(folder)
    print(files)
    path = folder + "/" + files[1]
    print("Image path: {}".format(path))
    img = cv2.imread(path)

    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Get the image and show it
    # fps = fps_counter.get_fps()


    

    height = img.shape[0]
    width = img.shape[1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = denoise_frame(img)

    # For thresholding
    # B G R
    low_th = (160, 160, 170)
    high_th = (255, 255, 255)
    
    frame_warped_colored = warp_perspective(img)

    mask = cv2.inRange(img, low_th, high_th)

    frame_roi = region_of_interest(mask)

    frame_canny = detect_edges(frame_roi)


    frame_warped = warp_perspective(frame_canny)



    # # l,r = histogram(frame)

    hough_lines=cv2.HoughLinesP(frame_warped, 1, np.pi/180 , 100, 
                                    np.array([]), minLineLength=20, maxLineGap=150)
    

    
    hough_lines_1 = cv2.HoughLinesP(frame_warped, rho=2, theta=np.pi/180,
                           threshold=20, lines=np.array([]), minLineLength=8, maxLineGap=5)

    optim_lines=optimize_lines(frame_warped,hough_lines)

    
    # frame_final=draw_lines(frame_warped,optim_lines)
    frame_final=display_lines(frame_warped_colored,optim_lines)

    # frame = cv2.addWeighted(frame, 1, mask, 1, 1)    


    # # frame=display_lines(frame_denoised_warped,optim_lines)
    cv2.imshow("img", frame_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(hough_lines)
    print("xxxxxxxxxxxx")
    print(optim_lines)

    


    # # draw fps on image
    # # cv2.putText(
    # #     frame,
    # #     f"FPS: {fps:.2f}",
    # #     (10, 30),
    # #     cv2.FONT_HERSHEY_SIMPLEX,
    # #     1,
    # #     (0, 255, 0),
    # #     2,
    # #     cv2.LINE_AA,
    # # )

    # cv2.imshow("image", frame)
    # # cv2.imshow("imh", img)
    # cv2.waitKey(1)


if __name__ == "__main__":
    # Initialize any variables needed
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # fps_counter = FPSCounter()

    run_car()

