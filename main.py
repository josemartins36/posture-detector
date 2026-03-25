import cv2
import mediapipe as mp

# inicializa mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

# webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # espelha imagem
    frame = cv2.flip(frame, 1)

    # converte para RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # processa pose
    results = pose.process(rgb)

    # desenha esqueleto
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # mostra na tela
    cv2.imshow("Pose Detector", frame)

    # sair com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
