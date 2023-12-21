import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modeli yükle
model = load_model("model240piksel.h5")

# Sınıf etiketlerini ve indekslerini eşleştiren bir sözlük
class_labels = {
    0: 'ace of clubs',
    1: 'ace of diamonds',
    2: 'ace of hearts',
    3: 'ace of spades',
    4: 'eight of clubs',
    5: 'eight of diamonds',
    6: 'eight of hearts',
    7: 'eight of spades',
    8: 'five of clubs',
    9: 'five of diamonds',
    10: 'five of hearts',
    11: 'five of spades',
    12: 'four of clubs',
    13: 'four of diamonds',
    14: 'four of hearts',
    15: 'four of spades',
    16: 'jack of clubs',
    17: 'jack of diamonds',
    18: 'jack of hearts',
    19: 'jack of spades',
    20: 'joker',
    21: 'king of clubs',
    22: 'king of diamonds',
    23: 'king of hearts',
    24: 'king of spades',
    25: 'nine of clubs',
    26: 'nine of diamonds',
    27: 'nine of hearts',
    28: 'nine of spades',
    29: 'queen of clubs',
    30: 'queen of diamonds',
    31: 'queen of hearts',
    32: 'queen of spades',
    33: 'seven of clubs',
    34: 'seven of diamonds',
    35: 'seven of hearts',
    36: 'seven of spades',
    37: 'six of clubs',
    38: 'six of diamonds',
    39: 'six of hearts',
    40: 'six of spades',
    41: 'ten of clubs',
    42: 'ten of diamonds',
    43: 'ten of hearts',
    44: 'ten of spades',
    45: 'three of clubs',
    46: 'three of diamonds',
    47: 'three of hearts',
    48: 'three of spades',
    49: 'two of clubs',
    50: 'two of diamonds',
    51: 'two of hearts',
    52: 'two of spades'
}

# Video yakalamayı başlat
cap = cv2.VideoCapture(0)  # Varsayılan kamerayı kullanarak video yakalamayı başlat
# Ekran boyutunu 240x240 olarak ayarla
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    # Görüntüyü yakala
    success, img = cap.read()

    # Görüntüyü işle ve modele giriş olarak uygun hale getir
    processed_img = cv2.resize(img, (240, 240))  # Görüntüyü yeniden boyutlandır
    processed_img = np.expand_dims(processed_img, axis=0)  # Boyutunu genişlet (1, 240, 240, 3)
    processed_img = processed_img / 255.0  # Görüntüyü [0, 1] aralığında normalize et

    # Sınıflandırmayı gerçekleştir
    predictions = model.predict(processed_img)
    class_index = np.argmax(predictions)
    class_label = class_labels[class_index]
    confidence = predictions[0, class_index]

    # Sonuçları görüntüye ekle
    text = f"Class: {class_label}, Confidence: {confidence:.2f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında döngüyü sonlandır
        break

# Kaynakları serbest bırak
cap.release()  # Video yakalamayı serbest bırak
cv2.destroyAllWindows()  # Tüm açık pencereleri kapat