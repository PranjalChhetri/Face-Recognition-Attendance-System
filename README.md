# Face-Recognition-Attendance-System

Final Year Face Recognition Attendance System Project.

![FACE DETECTION](https://user-images.githubusercontent.com/28294942/166667109-d2024d8c-9aec-44ed-93f8-8f1d9b66098a.png)

### Abstract 

The management of attendance can be a great burden on teachers if it is done by hand. To resolve this problem, a smart and auto attendance management system is being utilized. By utilizing this framework, the problem of proxies and students being marked present even though they are not physically present can easily be solved. 

This system marks attendance using a live video stream. The frames are extracted from the video using OpenCV. The main implementation steps used in this type of system are face detection and recognizing the detected face. After these steps, the recognized faces are compared with a database containing students' faces. This model will be a successful technique to manage the attendance of students.

Live Webcam based Face Attendance System Project through Python programming.

### Details :

Smart Attendance Management System is an application developed for daily student attendance in colleges or schools. This project attempts to record attendance through face detection.

This System uses facial recognition technology to record attendance through a high-resolution digital camera/webcam that detects and recognizes faces and compares the recognized faces with students/known faces' images stored in the faces database (CSV).

### Setup and Running

1. Create a folder named `Images_Attendance` in the root directory.
2. Place the images of the people you want to recognize in that folder.
3. Run the application:
   ```bash
   python AttendanceProject.py
   ```
4. A webcam window will appear and recognize faces in real-time, automatically logging them into `Attendance.csv`.
