import cv2
import os
import argparse
import mediapipe as mp


def process_img(img, face_detection ):
    img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    H,W,_ = img.shape

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1,y1,w,h = bbox.xmin,bbox.ymin , bbox.width , bbox.height

            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            
            #blur faces
            img[y1: y1+h , x1:x1+w , :] = cv2.blur(img[y1: y1+h , x1:x1+w , :]  , (40,40))
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode",default = 'webcam')
# args.add_argument("--filePath" ,  default='C:\\Users\\ritig\\Downloads\\photomy_out.jpg')  #for photo
# args.add_argument("--filePath" , default='C:\\Users\\ritig\\Downloads\\videomy.mp4') #for video path
args.add_argument("--filePath" , default=None)  #for webcam path

args = args.parse_args()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence = 0.5,model_selection=0) as face_detection:
    
    if args.mode in ['image']:
        #read image
        img = cv2.imread(args.filePath)

        

        img = process_img(img , face_detection)
        
        #save image
        cv2.imwrite('C:\\Users\\ritig\\Desktop\\openCV\\data\\photomy_out.jpg', img)

    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret,frame = cap.read()

        output_video  = cv2.VideoWriter('C:\\Users\\ritig\\Desktop\\openCV\\data\\videomy_out2.mp4', 
                                        cv2.VideoWriter_fourcc(*'MP$V'),
                                        25,
                                        (frame.shape[1] , frame.shape[0]))
        

        while ret:
            frame = process_img(frame , face_detection)

            output_video.write(frame)

            ret,frame = cap.read()

            

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret , frame = cap.read()

        while ret:
            frame = process_img(frame , face_detection)

            cv2.imshow('frame',frame)
            cv2.waitKey(30)

            ret,frame = cap.read()
        cap.release()


        
    
   


