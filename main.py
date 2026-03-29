import cv2
import mediapipe as mp
import numpy as np

# ===== MEDIAPIPE =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# ===== BASELINE =====
baseline = None

# ===== FUNÇÃO DE ÂNGULO =====
def calcular_angulo(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angulo = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    return angulo

# ===== WEBCAM =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    mensagens = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ===== PONTOS =====
        orelha_esq = [lm[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                      lm[mp_pose.PoseLandmark.LEFT_EAR.value].y]

        orelha_dir = [lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                      lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

        ombro_esq = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        ombro_dir = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        nariz = [lm[mp_pose.PoseLandmark.NOSE.value].x,
                 lm[mp_pose.PoseLandmark.NOSE.value].y]

        # ===== CENTROS =====
        centro_orelha = [(orelha_esq[0] + orelha_dir[0]) / 2,
                         (orelha_esq[1] + orelha_dir[1]) / 2]

        centro_ombro = [(ombro_esq[0] + ombro_dir[0]) / 2,
                        (ombro_esq[1] + ombro_dir[1]) / 2]

        # ===== MÉTRICAS =====
        delta_x = centro_orelha[0] - centro_ombro[0]
        delta_y = centro_orelha[1] - centro_ombro[1]
        diff_ombro = abs(ombro_esq[1] - ombro_dir[1])
        angulo_cabeca = calcular_angulo(centro_ombro, centro_orelha, nariz)

        print("dx:", delta_x, "angulo:", angulo_cabeca, "ombro:", diff_ombro)

        # ===== DETECÇÕES COM BASELINE =====
        if baseline:
            if delta_x < baseline["dx"] - 0.01:
                mensagens.append("CABECA A FRENTE")

            if angulo_cabeca < baseline["angulo"] - 3:
                mensagens.append("CABECA ABAIXADA")

            if diff_ombro > baseline["ombro"] + 0.02:
                mensagens.append("OMBROS DESALINHADOS")

        # ===== DESENHO =====
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        h, w, _ = frame.shape

        px1 = int(centro_orelha[0] * w)
        py1 = int(centro_orelha[1] * h)
        px2 = int(centro_ombro[0] * w)
        py2 = int(centro_ombro[1] * h)

        cv2.circle(frame, (px1, py1), 6, (255, 0, 0), -1)
        cv2.circle(frame, (px2, py2), 6, (255, 0, 0), -1)

        cv2.line(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)

        # ===== DEBUG =====
        cv2.putText(frame, f"dx: {delta_x:.3f}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"ang: {angulo_cabeca:.1f}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"ombro: {diff_ombro:.3f}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ===== TECLAS =====
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and results.pose_landmarks:
        baseline = {
            "dx": delta_x,
            "angulo": angulo_cabeca,
            "ombro": diff_ombro
        }
        print("CALIBRADO:", baseline)

    # ===== TEXTO =====
    if baseline is None:
        cv2.putText(frame, "Pressione C para calibrar",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)
    else:
        if mensagens:
            for i, msg in enumerate(mensagens):
                cv2.putText(frame, msg,
                            (30, 50 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2)
        else:
            cv2.putText(frame, "POSTURA OK",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

    cv2.imshow("Detector de Postura (CALIBRADO)", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
