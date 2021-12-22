# IMPORTING LIBRARIES
import cv2
import mediapipe as mp
import numpy as np
import math

CEF_COUNTER = 0
CLOSED_EYES_FRAME = 2
TOTAL_BLINKS = 0
FONTS =cv2.FONT_HERSHEY_COMPLEX
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)
left_count = 0
right_count = 0
flag = 0

face_detection_flag = 5
face_detected = 0
frame_count = 0
movement_detection_flag = 6
first_two_step = 0

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[BLACK, GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [YELLOW, PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [GRAY, YELLOW]
    else:
        pos_eye="Closed"
        color = [GRAY, YELLOW]
    return pos_eye, color

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv2.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 


def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background 
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img

# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def fillPolyTrans(img, points, color, opacity):
    """
    @param img: (mat) input image, where shape is drawn.
    @param points: list [tuples(int, int) these are the points custom shape,FillPoly
    @param color: (tuples (int, int, int)
    @param opacity:  it is transparency of image.
    @return: img(mat) image with rectangle draw.

    """
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()  # coping the image
    cv2.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
    return img

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#EYE CO-ORDINATES
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# DETECT THE FACE LANDMARKS
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  while True:
    success, image = cap.read()

    # Flip the image horizontally and convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image_height, image_width= image.shape[:2]

    # To improve performance
    image.flags.writeable = False
    
    # Detect the face landmarks
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True

    # Convert back to the BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
    
    # Draw the face mesh annotations on the image.
    if results.multi_face_landmarks:
        
      frame_count = frame_count + 1
      if frame_count == 30 :
        #print(face_detection_flag)
        face_detection_flag = face_detection_flag - 1
        if face_detection_flag < -3:
            movement_detection_flag = movement_detection_flag - 1
            frame_count = 1
        frame_count = 1
      if face_detection_flag >= 0:
          cv2.putText(image, f'{face_detection_flag}', (100, 50), FONTS, 1.5, GREEN, 2)
      if face_detection_flag < 0 and face_detection_flag >= -3:
          cv2.putText(image, "FACE DETECTED", (100, 50), FONTS, 1.5, GREEN, 2)
          face_detected = 1          
          
      if movement_detection_flag >= 0 and face_detection_flag < -3:
          cv2.putText(image, f'{movement_detection_flag}', (100, 50), FONTS, 1.5, GREEN, 2)
      if movement_detection_flag < 0 and movement_detection_flag >= -3:
          cv2.putText(image, "MOVEMENT DETECTED", (100, 50), FONTS, 1.5, GREEN, 2)
          first_two_step = 1
        
      mesh_coords = landmarksDetection(image, results, False)
      ratio = blinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
      
      if ratio > 4.8 and first_two_step == 1:
        CEF_COUNTER +=1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
        colorBackgroundText(image,  f'Blink', FONTS, 1.7, (int(image_height/2), 100), 2, (0,255,255), pad_x=6, pad_y=6, )

      else:
        if CEF_COUNTER>CLOSED_EYES_FRAME:
          TOTAL_BLINKS +=1
          CEF_COUNTER =0
            # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
      colorBackgroundText(image,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
      image = fillPolyTrans(image, [mesh_coords[p] for p in LEFT_EYE], (0,255,0), opacity=0.4)
      image = fillPolyTrans(image, [mesh_coords[p] for p in RIGHT_EYE], (0,255,0), opacity=0.4)
      
      # Blink Detector Counter Completed
      right_coords = [mesh_coords[p] for p in RIGHT_EYE]
      left_coords = [mesh_coords[p] for p in LEFT_EYE]
      crop_right, crop_left = eyesExtractor(image, right_coords, left_coords)
      eye_position, color = positionEstimator(crop_right)
      #colorBackgroundText(image, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
      eye_position_left, color = positionEstimator(crop_left)
      #colorBackgroundText(image, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
      if eye_position == "CENTER" and eye_position_left == "CENTER" and flag == 1 and first_two_step == 1:
        flag = 0
      if eye_position == "RIGHT" and eye_position_left == "RIGHT" and flag == 0 and first_two_step == 1:
        flag = 1
        left_count = left_count + 1
      if eye_position == "LEFT" and eye_position_left == "LEFT" and flag == 0 and first_two_step == 1:
        flag = 1
        right_count = right_count + 1
      colorBackgroundText(image, f'Left Movement: {left_count}', FONTS, 0.5, (40, 220), 2, color[0], color[1], 8, 8)
      colorBackgroundText(image, f'Right Movement: {right_count}', FONTS, 0.5, (40, 320), 2, color[0], color[1], 8, 8)

      if left_count >= 3 and right_count >= 3 and TOTAL_BLINKS >=3 and first_two_step == 1:
        cv2.putText(image, "VERIFICATION DONE", (100, 50), FONTS, 1.5, GREEN, 2)
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
    else:
        left_count = 0
        right_count = 0
        TOTAL_BLINKS = 0
        face_detection_flag = 5
        movement_detection_flag = 6
        face_detected = 0
        first_two_step = 0

    # Display the image
    cv2.imshow('MediaPipe FaceMesh', image)
    
    # Terminate the process
    if cv2.waitKey(5) & 0xFF == 27:
      cv2.destroyAllWindows()
      break

cap.release()
