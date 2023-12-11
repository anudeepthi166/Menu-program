import subprocess
import speech_recognition as sr
import pyfiglet
import webbrowser
import datetime
import cv2
import os
import face_recognition
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dictionary to store student information
student_info = {
    "O170111": {
        "name": "Anuhya Velagaturi",
        "id_number": "O170111",
        "branch": "CSE",
        "year": 4,
        "batch": "2017-2023",
        "image_path": "/root/anu/anuhya.jpg"
    },
    "O170122": {
        "name": "Vijaya Themmanaboina",
        "id_number": "O170122",
        "branch": "CSE",
        "year": 4,
        "batch": "2017-2023",
        "image_path": "/root/anu/vijaya.jpg"
    },
    "O170133": {
        "name": "Kamakshi Godini",
        "id_number": "O170133",
        "branch": "CSE",
        "year": 4,
        "batch": "2017-2023",
        "image_path": "/root/anu/kamakshi.jpg"
    }
    "O170144": {
        "name": "Anudeepthi Kolagani",
        "id_number": "O170144",
        "branch": "CSE",
        "year": 4,
        "batch": "2017-2023",
        "image_path": "/root/anu/deepthi.jpg"
    }
    # Add more student information as needed
}

# Dictionary to store known face encodings and id numbers
known_faces = {}


# Function to load and encode a student face
def load_student_face(id_number, image_path):
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces[id_number] = face_encoding


# Load known faces and their id numbers
for id_number, student in student_info.items():
    image_path = student["image_path"]
    if os.path.exists(image_path):
        load_student_face(id_number, image_path)
    else:
        print(f"Image file not found for id number: {id_number}")


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something!")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand your voice.")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""


def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output)


def display_header():
    header = pyfiglet.figlet_format("MENU PROGRAM", font="slant")
    print(header)


def detect_eyes():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def match_face(face_encoding):
    for id_number, known_face_encoding in known_faces.items():
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        if matches[0]:
            return id_number
    return None


def ml_interface():
    print("Hors of study:")
    x = pd.read_csv('data.csv')
    a = x["hrs of study"]
    b = np.array(a)
    b = b.reshape(-1, 1)
    y = x["marks secured"]
    mind = LinearRegression()
    mind.fit(b, y)
    study_hours = int(input())
    marks_predicted = mind.predict([[study_hours]])
    print("Predicted marks:", marks_predicted[0])


def main():
    display_header()
    print("Please choose any of the following options:")
    print("1. Type manually")
    print("2. Voice recognition")
    choice = input("Please enter your choice: ")

    while True:
        if choice == "1":
            print("""
            Press a: for attendance face detection
            Press e: for eye detection
            Press l: for Linux commands
            Press ML: to run ML code
            Press google: to search on Google
            Press youtube: to open YouTube
            Press time: to get current time
            """)
            command = input("Enter your choice: ")
            if command == "google":
                search_term = input("You want to search for? ")
                url = "https://www.google.com/search?q=" + search_term
                webbrowser.open_new_tab(url)
            elif command == "youtube":
                webbrowser.open_new_tab("https://www.youtube.com/")
            elif command == "time":
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print("The current time is:", current_time)
            elif command == "y":
                detect_eyes()
            elif command == "a":
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                video_capture = cv2.VideoCapture(0)

                while True:
                    ret, frame = video_capture.read()

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_color = frame[y:y + h, x:x + w]

                        face_encoding = face_recognition.face_encodings(frame)[0]
                        id_number = match_face(face_encoding)

                        if id_number:
                            student = student_info[id_number]
                            cv2.rectangle(frame, (10, 10), (500, 220), (200, 200, 200), cv2.FILLED)
                            cv2.putText(frame, "MATCHED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            cv2.putText(frame, f"Name: {student['name']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 0), 2)
                            cv2.putText(frame, f"ID No: {student['id_number']}", (20, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            cv2.putText(frame, f"Branch: {student['branch']}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 0), 2)
                            cv2.putText(frame, f"Year: {student['year']}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 0), 2)
                            cv2.putText(frame, f"Batch: {student['batch']}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 0), 2)

                    cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()

            elif command == "ML":
                ml_interface()
            elif command == "l":
                while True:
                    print("""
                    Press date: to run date command
                    Press docker: to run docker commands
                    Press cal: to print calendar
                    Press docker_images: to list docker images
                    Press ls: to list files
                    Press cd: to change directory
                    Press whoami: to get current user
                    Press docker_version: to get Docker version
                    """)
                    subcommand = input("Enter your choice: ")
                    if subcommand == "docker_images":
                        run_command("docker images")
                    elif subcommand == "docker_version":
                        run_command("docker --version")
                    elif subcommand == "ls":
                        run_command("ls")
                    elif subcommand == "cd":
                        directory = input("Enter directory path: ")
                        command = f"cd {directory}"
                        run_command(command)
                    elif subcommand == "cal":
                        run_command("cal")
                    elif subcommand == "whoami":
                        run_command("whoami")
                    elif subcommand == "date":
                        run_command("date")
                    elif subcommand == "docker":
                        print("""
                        Press list: to list all containers
                        Press rc: to list running containers
                        Press sc: to list stopped containers
                        """)
                        docker_subcommand = input("Enter your choice: ")
                        if docker_subcommand == "1":
                            run_command("docker ps -a")
                        elif docker_subcommand == "2":
                            run_command("docker ps")
                        elif docker_subcommand == "3":
                            run_command("docker ps -f status=exited")
                        else:
                            print("Invalid choice")
                    else:
                        print("Invalid choice")

                    answer = input("Is there anything else I can do for you? (yes/no): ")
                    if answer.lower() != "yes":
                        break
            else:
                print("Invalid choice")
        elif choice == "2":
            command = get_audio()
            if command:
                run_command(command)
        else:
            print("Invalid choice")

        answer = input("Is there anything else I can do for you? (yes/no): ")
        if answer.lower() != "yes":
            break


if __name__ == "__main__":
    main()
