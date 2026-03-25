import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)


def calcular_angulo(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.degrees(np.arccos(cos))

    return angulo


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    mensagem = "POSTURA OK"

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        # pontos principais

        orelha_esq = [
            lm[mp_pose.PoseLandmark.LEFT_EAR.value].x,
            lm[mp_pose.PoseLandmark.LEFT_EAR.value].y
        ]

        orelha_dir = [
            lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
            lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y
        ]

        ombro_esq = [
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]

        ombro_dir = [
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]

        quadril_esq = [
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]

        quadril_dir = [
            lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ]

        # centros

        centro_orelha = [
            (orelha_esq[0] + orelha_dir[0]) / 2,
            (orelha_esq[1] + orelha_dir[1]) / 2
        ]

        centro_ombro = [
            (ombro_esq[0] + ombro_dir[0]) / 2,
            (ombro_esq[1] + ombro_dir[1]) / 2
        ]

        centro_quadril = [
            (quadril_esq[0] + quadril_dir[0]) / 2,
            (quadril_esq[1] + quadril_dir[1]) / 2
        ]

        # ===== DETECTOR DE CABEÇA =====

        if centro_orelha[0] < centro_ombro[0] - 0.08:
            mensagem = "CABECA MUITO A FRENTE"

        # ===== DETECTOR DE COLUNA =====

        angulo_coluna = calcular_angulo(
            centro_orelha,
            centro_ombro,
            centro_quadril
        )

        if angulo_coluna < 160:
            mensagem = "COLUNA CURVADA"

        # ===== DETECTOR DE OMBROS =====

        if abs(ombro_esq[1] - ombro_dir[1]) > 0.05:
            mensagem = "OMBROS DESALINHADOS"

        # ===== DESENHAR PONTOS =====

        h, w, _ = frame.shape

        for ponto in [centro_orelha, centro_ombro, centro_quadril]:

            px = int(ponto[0] * w)
            py = int(ponto[1] * h)

            cv2.circle(frame, (px, py), 8, (255, 0, 0), -1)

        # desenhar esqueleto

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # ===== TEXTO NA TELA =====

    if mensagem == "POSTURA OK":
        cor = (0, 255, 0)
    else:
        cor = (0, 0, 255)

    cv2.putText(
        frame,
        mensagem,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        cor,
        3
    )

    cv2.imshow("Detector de Postura", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
