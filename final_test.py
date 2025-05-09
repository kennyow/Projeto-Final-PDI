import tensorflow as tf
import numpy as np
import cv2
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Carrega modelo treinado e cria um modelo auxiliar para extrair features
base_model = tf.keras.models.load_model("modelo_cifar10_otimizado.keras")
feature_extractor = tf.keras.Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output  # Extrai features da penúltima camada
)
class_names = ['aviao', 'carro', 'passaro', 'gato', 'cervo',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao']

def is_center_red(hsv, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2
    pixel = hsv[cy, cx]
    return (
        (0 <= pixel[0] <= 10 or 160 <= pixel[0] <= 180) and
        pixel[1] >= 100 and
        pixel[2] >= 100
    )

def webcam_classification():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    print("Pressione 'q' para sair.")

    active_boxes = {}  # chave: (x_round, y_round) → valor: (x, y, w, h, label, features)
    class_history = defaultdict(list)
    similarity_threshold = 0.9  # Limiar de similaridade para considerar imagens iguais

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Máscara para tons de vermelho
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_squares = []
        detected_positions = set()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                pos_key = (x // 20, y // 20)

                if is_center_red(hsv, x, y, w, h):
                    red_squares.append((x, y, w, h))
                    detected_positions.add(pos_key)
                else:
                    roi = frame[y:y + h, x:x + w]
                    if roi.size == 0:
                        continue

                    resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
                    normalized = resized.astype('float32') / 255.0

                    # Extrai features da imagem (vetor de características)
                    features = feature_extractor.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
                    
                    # Faz predição da classe
                    pred = base_model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
                    class_id = np.argmax(pred)
                    confidence = np.max(pred)

                    # Atualiza histórico
                    class_history[pos_key].append(class_id)
                    if len(class_history[pos_key]) > 5:
                        class_history[pos_key].pop(0)

                    if class_history[pos_key]:
                        final_class_id = max(set(class_history[pos_key]), key=class_history[pos_key].count)
                        label = f"{class_names[final_class_id]} ({confidence*100:.0f}%)"
                        active_boxes[pos_key] = (x, y, w, h, label, features)

                    detected_positions.add(pos_key)

        # Remove boxes de posições que voltaram a ser vermelhas
        keys_to_remove = [key for key in active_boxes if key not in detected_positions]
        for key in keys_to_remove:
            del active_boxes[key]
            if key in class_history:
                del class_history[key]

        # Detecta pares de imagens iguais (com base nas features)
        boxes_list = list(active_boxes.values())
        matched_pairs = set()
        
        for i in range(len(boxes_list)):
            for j in range(i + 1, len(boxes_list)):
                # Compara similaridade de cosseno entre as features
                sim = cosine_similarity(
                    [boxes_list[i][5]],  # features do box i
                    [boxes_list[j][5]]   # features do box j
                )[0][0]
                if sim > similarity_threshold:
                    matched_pairs.add(i)
                    matched_pairs.add(j)

        # Exibe os boxes (verde para únicos, azul para pares iguais)
        for idx, (x, y, w, h, label, _) in enumerate(boxes_list):
            color = (0, 255, 0)  # Verde padrão
            if idx in matched_pairs:
                color = (255, 0, 0)  # Azul para imagens iguais
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Quadrados vermelhos: {len(red_squares)} | Objetos: {len(active_boxes)}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Jogo da Memória - CIFAR10", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

webcam_classification()