__author__ = 'Vaibhav Punia'

import sys
from PyQt4 import QtGui
from PyQt4 import QtCore
import cv2
import os
import numpy as np
from PIL import Image


def window():
    app = QtGui.QApplication(sys.argv)
    w = QtGui.QWidget()

    w.setWindowTitle("Face Recognotion")
    w.setGeometry(400,200,610,300)

    b = QtGui.QLabel(w)
    b.setText("Face Recognotion")
    b.setStyleSheet("color: #ff1a1a")
    b.move(220,30)
    titlefont = QtGui.QFont("Times", 20, QtGui.QFont.Bold)
    b.setFont(titlefont)

    labell = QtGui.QLabel(w)
    labell.setText("Enter your name:")
    labell.move(150,105)
    labelfont = QtGui.QFont("Times", 12, QtGui.QFont.Bold)
    labell.setFont(labelfont)

    global textbox

    textbox = QtGui.QLineEdit(w)
    textbox.move(280, 105)
    textbox.resize(200, 25)

    ex_btn = QtGui.QPushButton(w)
    ex_btn.setText("Detect Face")
    ex_btn.resize(100,60)
    ex_btn.setStyleSheet('QPushButton {color: blue; font-size: 12px}')
    ex_btn.move(150,200)
    ex_btn.clicked.connect(face_detection)

    ex_btn = QtGui.QPushButton(w)
    ex_btn.setText("Recognize Face")
    ex_btn.resize(100, 60)
    ex_btn.setStyleSheet('QPushButton {color: blue; font-size: 12px}')
    ex_btn.move(350, 200)
    ex_btn.clicked.connect(face_recognition)

    ex_btn = QtGui.QPushButton(w)
    ex_btn.setText("Get data and train")
    ex_btn.resize(110, 40)
    ex_btn.setStyleSheet('QPushButton {color: blue; font-size: 12px}')
    ex_btn.move(490, 100)
    ex_btn.clicked.connect(get_data)



    global label
    label = QtGui.QLabel(w)
    #label.setText("Error")
    label.setText("                                                                           ")
    label.setStyleSheet("font-size: 12px")
    label.move(200,170)

    w.show()
    sys.exit(app.exec_())

def get_data():
    #f = open("Face Log.txt", "w")
    cascadePath = "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(cascadePath)
    cam = cv2.VideoCapture(0)
    Id = 1
    sampleNum = 0

    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite("dataSet/User." + str(Id) + '.' + str(sampleNum) + ".jpg", gray)

            cv2.imshow('frame', img)

            # time.sleep(10)
        # wait for 100 miliseconds
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 20
        elif sampleNum > 20:
            break

    cam.release()
    cv2.destroyAllWindows()

    recognizer = cv2.createLBPHFaceRecognizer()

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        # get the path of all the files in the folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # create empth face list
        faceSamples = []
        # create empty ID list
        Ids = []
        # now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces = detector.detectMultiScale(imageNp)
            # If a face is there then append that in the list as well as Id of it
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
        return faceSamples, Ids

    faces, Ids = getImagesAndLabels("dataSet")
    recognizer.train(faces, np.array(Ids))
    recognizer.save('trainer/trainer.yml')

def face_detection():
    global textbox

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, "Press s to capture", (0, 10), font, 0.55, (255, 0, 0), 2)

        cv2.imshow("im", img)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            still_im = img
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gray = cv2.cvtColor(still_im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(still_im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(still_im, str(textbox.text()), (x, y - 10), font, 1, (0, 255, 0), 2)

    cv2.imshow("im", still_im)
    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

def face_recognition():
    recognizer = cv2.createLBPHFaceRecognizer()
    recognizer.load('trainer/trainer.yml')

    f = open("Face Log.txt", "r")

    data = f.readlines()

    d = {}

    for i in data:
        id, name = i.split(":")
        d[id] = name

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, "Press s to capture", (0, 10), font, 0.55, (255, 0, 0), 2)

        cv2.imshow("im", img)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            still_im = img
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    font = cv2.FONT_HERSHEY_COMPLEX

    gray = cv2.cvtColor(still_im, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(still_im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        print conf, Id
        if conf < 60:
            cv2.putText(still_im, "Matched", (x, y - 10), font, 1, (0, 255, 0), 2)

            user_path = "dataset\\User." + str(Id) + ".1.jpg"
            predicted_user = cv2.imread(user_path)
            predicted_user_gray = cv2.cvtColor(predicted_user, cv2.COLOR_BGR2GRAY)
            predicted_user_faces = faceCascade.detectMultiScale(predicted_user_gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(predicted_user, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Predected User.", predicted_user)
            cv2.imshow("IM", still_im)
            cv2.waitKey(0)
        else:
            cv2.putText(still_im, "No Match", (x, y - 10), font, 1, (0, 0, 255), 2)
            cv2.rectangle(still_im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            user_path = "dataset\\User." + str(Id) + ".1.jpg"
            predicted_user = cv2.imread(user_path)
            predicted_user_gray = cv2.cvtColor(predicted_user, cv2.COLOR_BGR2GRAY)
            predicted_user_faces = faceCascade.detectMultiScale(predicted_user_gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(predicted_user, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Predected User.", predicted_user)
            cv2.imshow("IM", still_im)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    window()
