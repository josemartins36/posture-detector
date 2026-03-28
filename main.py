import cv2
import mediapipe as mp
import numpy as np

# ===== MEDIAPIPE =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# ===== DETECÇÕES =====
def detectar_cabeca_frente(orelha, ombro):
    delta_x = orelha[0] - ombro[0]
    print("delta_x (frente):", delta_x)

    if delta_x < -0.08:
        return "CABECA A FRENTE"
    return None

def detectar_cabeca_baixo(orelha, ombro):
    delta_y = orelha[1] - ombro[1]
    print("delta_y (baixo):", delta_y)

    if delta_y < -0.15:
        return "CABECA ABAIXADA"
    return None


def detectar_ombros(ombro_esq, ombro_dir):
    diff = abs(ombro_esq[1] - ombro_dir[1])
    print("diff ombro:", diff)

    if diff > 0.03:
        return "OMBROS DESALINHADOS"
    return None

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

        # ===== CENTROS =====

        centro_orelha = [
            (orelha_esq[0] + orelha_dir[0]) / 2,
            (orelha_esq[1] + orelha_dir[1]) / 2
        ]

        centro_ombro = [
            (ombro_esq[0] + ombro_dir[0]) / 2,
            (ombro_esq[1] + ombro_dir[1]) / 2
        ]

        # ===== DETECÇÕES =====

        res1 = detectar_cabeca_frente(centro_orelha, centro_ombro)
        res2 = detectar_cabeca_baixo(centro_orelha, centro_ombro)
        res3 = detectar_ombros(ombro_esq, ombro_dir)

        if res1:
            mensagens.append(res1)

        if res2:
            mensagens.append(res2)

        if res3:
            mensagens.append(res3)

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

        # pontos
        cv2.circle(frame, (px1, py1), 6, (255, 0, 0), -1)
        cv2.circle(frame, (px2, py2), 6, (255, 0, 0), -1)

        # linha cabeça → ombro
        cv2.line(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)

        # ===== DEBUG NA TELA =====

        delta_x = centro_orelha[0] - centro_ombro[0]
        delta_y = centro_orelha[1] - centro_ombro[1]
        diff_ombro = abs(ombro_esq[1] - ombro_dir[1])

        cv2.putText(frame, f"dx: {delta_x:.3f}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"dy: {delta_y:.3f}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"ombro: {diff_ombro:.3f}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ===== TEXTO =====

    if mensagens:
        for i, msg in enumerate(mensagens):
            cv2.putText(
                frame,
                msg,
                (30, 50 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
    else:
        cv2.putText(
            frame,
            "POSTURA OK",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Detector de Postura (DEBUG)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
