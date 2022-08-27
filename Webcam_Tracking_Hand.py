import cv2
import mediapipe as mp


mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils



cap=cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        succes,image=cap.read()
        if not succes:
            print("camera not opened")
            continue
       
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=cv2.flip(image,1)
        
        image=cv2.resize(image,(950,580))

        results=hands.process(image)

        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS)
                print(results.multi_handedness)
        
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()