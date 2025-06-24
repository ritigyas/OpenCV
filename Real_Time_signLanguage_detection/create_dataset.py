#create dataset
import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True , min_detection_confidence=0.3)

DATA_DIR='./data'
data=[]
labels=[]

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux=[]
        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
        img_rgb=cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        if img is None:
                    print(f"Could not read image: {img_path}")
                    continue



        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21): 
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
        
        if len(data_aux) == 42:  
            data.append(data_aux)
            labels.append(int(dir_))
  

                

f=open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()

from collections import Counter
print(Counter(labels))



            
            

        

#         plt.figure()
#         plt.imshow(img_rgb)
# plt.show()



# for dir_ in os.listdir(DATA_DIR):
#     print(f"Processing class: {dir_}")
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         img_full_path = os.path.join(DATA_DIR, dir_, img_path)
#         img = cv2.imread(img_full_path)

#         if img is None:
#             print(f"⚠️ Could not read image: {img_full_path}")
#             continue

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)

#         if not results.multi_hand_landmarks:
#             print(f"❌ No hands detected in: {img_full_path}")
#             continue

#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 img_rgb,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#         plt.figure()
#         plt.title(f"{dir_}/{img_path}")
#         plt.imshow(img_rgb)

# plt.show()

# print(f"Dir: {dir_} → First 4 landmarks: {data_aux[:8]}")
