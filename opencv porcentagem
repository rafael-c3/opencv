import cv2
import numpy as np

# Carrega o modelo de reconhecimento facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega o modelo de classificação de gênero
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# Função para detectar e classificar o gênero
def detect_gender(frame, face):
    # Extrai a região de interesse (ROI) do rosto
    (x, y, w, h) = face
    face_img = frame[y:y+h, x:x+w].copy()

    # Pré-processamento da imagem para a entrada do modelo de classificação de gênero
    blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Passa a imagem pela rede para obter as previsões de gênero
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()

    # Obtém o índice da previsão de gênero com maior probabilidade
    gender = ['Male', 'Female'][gender_preds[0].argmax()]
    # Obtém a porcentagem de ser homem e mulher
    percentage_male = gender_preds[0][0] * 100
    percentage_female = gender_preds[0][1] * 100

    return gender, percentage_male, percentage_female

# Inicializa a webcam
cap = cv2.VideoCapture(0)

while True:
    # Captura um frame
    ret, frame = cap.read()

    # Converte o frame para escala de cinza para o reconhecimento facial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Para cada rosto detectado, desenha um retângulo ao redor dele e classifica o gênero
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gender, percentage_male, percentage_female = detect_gender(frame, (x, y, w, h))
        cv2.putText(frame, f'{gender} ({percentage_male:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.putText(frame, f'Female ({percentage_female:.2f}%)', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # Mostra o frame resultante
    cv2.imshow('Gender Detection', frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
