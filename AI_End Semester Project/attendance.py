import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "Images"
images_list = []
student_names_list = []
myList = os.listdir(path)

for student_name in myList:
    current_img = cv2.imread(f"{path}/{student_name}")
    images_list.append(current_img)
    student_names_list.append(os.path.splitext(student_name)[0])


def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = face_recognition.face_encodings(img)[0]
        encode_list.append(encode_img)
    return encode_list


def mark_attendance(name):
    with open("Attendance.csv", "r+") as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(",")
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{date_string}")


# calling the function to encode the list of images
encode_List_Known = find_encodings(images_list)
print("Encoding Completed")
# Starting Webcam
capture = cv2.VideoCapture(0)


def resolution_720p():
    capture.set(3, 1280)
    capture.set(4, 720)


resolution_720p()

while True:
    success, img = capture.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    face_current_Frames = face_recognition.face_locations(img_small)
    encode_current_Frames = face_recognition.face_encodings(img_small, face_current_Frames)

    for encode_Face, Face_Location in zip(encode_current_Frames, face_current_Frames):
        matches = face_recognition.compare_faces(encode_List_Known, encode_Face)
        face_distance = face_recognition.face_distance(encode_List_Known, encode_Face)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = student_names_list[match_index].upper()
            y1, x2, y2, x1 = Face_Location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 87, 51), 2)
            cv2.putText(img, name, (x1 - 25, y2 + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
