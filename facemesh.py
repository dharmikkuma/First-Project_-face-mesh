
import cv2
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(False,3)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

def findFaceMesh(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =faceMesh.process(imgRGB)
    faces = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,drawSpec, drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                            # 0.7, (0, 255, 0), 1)

                    #print(id,x,y)
                    face.append([x,y])
            faces.append(face)
    return img, faces

def main():
    cap = cv2.VideoCapture("E:\\Coursera\\Computer Vision\\OpenCV\\FACENET\\1.mp4")
    
    pTime = 0
    while True:
        success, img = cap.read()
        img, faces = findFaceMesh(img)

        if len(faces)!= 0:
            # print(faces[0])
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
            cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
main()