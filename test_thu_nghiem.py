import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import playsound
from threading import Thread
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

model = load_model("model_train_da_xong.h5")

def sound_alarm(path):
    playsound.playsound(path)

isWarning = False

while True:
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in face:
        face_img = img[y : y + h, x : x + w]
        cv2.imwrite('Hienthi.jpg', face_img)
        test_image = image.load_img('Hienthi.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        predict_image = model.predict(test_image)[0][0]
        print(model.predict(test_image)[0][0])

        if predict_image == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 3)
            cv2.putText(img, 'KHONG DEO KHAU TRANG', ((x+w) // 2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 1, 25), 3)
            if isWarning == False:
                t = Thread(target=sound_alarm, args=('chuong_bao_dong_351125.mp3',))
                t.start()
                isWarning = True

            if t.is_alive() == False:
                isWarning = False
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, 'CO DEO KHAU TRANG', ((x+w) // 2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 1, 25), 3)



    cv2.imshow('Nguyen Xuan Hieu', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
