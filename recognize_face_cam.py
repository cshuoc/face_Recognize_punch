import tkinter as tk
from tkinter import simpledialog
import argparse
import csv
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
import cv2
import datetime
import os
import pandas as pd
from PIL import Image, ImageTk

def save_face_database(face_database, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'embedding'])
        for label, data in face_database.items():
            embedding_str = ','.join(map(str, data['embedding'].tolist()))
            writer.writerow([label, embedding_str])

def load_face_database(csv_file_path):
    face_database = {}
    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = row['label']
                embedding = np.array(list(map(float, row['embedding'].split(','))))
                face_database[label] = {'embedding': torch.tensor(embedding), 'label': label}
    except FileNotFoundError:
        print(f"No database found at {csv_file_path}, starting a new database.")
    return face_database
   
def punch_in_log(punch_data_path, name):
    global df
    now = datetime.datetime.now()
    current_time = now.strftime('%H:%M')
    is_morning = now.hour < 12  # 判斷是否是上午
    today_date = now.strftime('%Y-%m-%d')
    mon_date = now.strftime('%Y_%m')

    # 讀取CSV文件
    empolyee_file_path = punch_data_path + '/' + name
    if not os.path.exists(empolyee_file_path):
        os.makedirs(empolyee_file_path)

    log_file_path = empolyee_file_path + '/' + name + '_' + str(mon_date) + '.csv'#
    
    print(empolyee_file_path)    
    try:
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Date', 'punch in', 'punch out'])
        df.to_csv(log_file_path, index=False)

    # 檢查是否已有當日記錄
    record = df[df['Date'] == today_date]
    
    if record.empty:
        # 如果沒有當日記錄，則新增
        new_record = {'Date': today_date, 'punch in': '', 'punch out': ''}
        df = df._append(new_record, ignore_index=True)
    else:
        # 如果已有當日記錄，則根據需要更新
        if is_morning and pd.isna(record['punch in'].values[0]):
            df.loc[df['Date'] == today_date, 'punch in'] = current_time
        if not is_morning and pd.isna(record['punch out'].values[0]):
            df.loc[df['Date'] == today_date, 'punch out'] = current_time
    
    # 保存到CSV
    df.to_csv(log_file_path, index=False)  
    return True  

def run_face_recognition(args, input_id):
    # 初始化 InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    csv_file_path = args.face_csv_path

    # 载入人脸数据库
    face_database = load_face_database(csv_file_path)

    if input_id not in face_database:
        response = tk.messagebox.askquestion("新增臉孔", f"我不認識你，是否新增臉孔？")
        if response == 'yes':
            add_new_face(input_id, face_cascade, model, face_database, csv_file_path)
    else:
        cap = cv2.VideoCapture(0)
        entry = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame from camera.")
                break
            faces = face_cascade.detectMultiScale(frame, 1.3, 2)
            if not entry:
                cv2.putText(frame, "Recognising...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "PUNCH IN, Please press ENTER to leave", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 225, 0), 2)
                cv2.putText(frame, "Good morning " + input_id, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 0), 2)
                
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_area = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_area,1.3,10)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_area, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                results = recognize_face_from_frame(frame, face_cascade, model, face_database, threshold=0.8)
                if results == input_id:  # 檢查辨識結果是否與輸入的工號匹配
                    punch_in = punch_in_log(args.log_file_path, input_id)
                    if punch_in:
                        entry = True
                        
            cv2.imshow('frame2',frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:    
                break

    cap.release()
    cv2.destroyAllWindows()

def add_new_face_from_frame(frame, label, face_cascade, model, face_database, csv_file_path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        img_resized = img.resize((256,256))
        img_cropped = np.array(img_resized)
        img_cropped = np.transpose(img_cropped, (2, 0, 1))
        img_tensor = torch.tensor(img_cropped).unsqueeze(0).float() / 255.0
        embedding = model(img_tensor).detach().cpu().squeeze()
        face_database[label] = {'embedding': embedding, 'label': label}
        save_face_database(face_database, csv_file_path)

def add_new_face(input_id, face_cascade, model, face_database, csv_file_path):
    cap = cv2.VideoCapture(0)
    face_added = False
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame from camera.")
            break
        faces = face_cascade.detectMultiScale(frame, 1.3, 2)
        if not face_added:
            cv2.putText(frame, "PRESS ENTER TO SAVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Successful Entry", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_area = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_area,1.3,10)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_area, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.imshow('Add New Face', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 13:  # 按下 ENTER 鍵
            add_new_face_from_frame(frame, input_id, face_cascade, model, face_database, csv_file_path)
            face_added = True
            
    cap.release()
    cv2.destroyAllWindows()

def recognize_face_from_frame(frame, face_cascade, model, face_database, threshold=0.85):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        img_resized = img.resize((256,256))
        img_cropped = np.array(img_resized)
        img_cropped = np.transpose(img_cropped, (2, 0, 1))
        img_tensor = torch.tensor(img_cropped).unsqueeze(0).float() / 255.0
        embedding = model(img_tensor).detach().cpu().squeeze()

        highest_sim = 0
        min_label = None
        for label, data in face_database.items():
            known_embedding = data['embedding']
            sim = similarity(embedding, known_embedding)
            if sim > highest_sim:
                highest_sim = sim
                min_label = label

        if highest_sim > threshold:
            return face_database[min_label]['label']

def similarity(embedding1, embedding2):
    embedding1 = embedding1.view(1, -1)
    embedding2 = embedding2.view(1, -1)
    cosine_sim = F.cosine_similarity(embedding1, embedding2).item()
    euclidean_dist = torch.norm(embedding1 - embedding2).item()
    euclidean_sim = 1 - euclidean_dist / 10.0
    combined_sim = 0.5 * cosine_sim + 0.5 * euclidean_sim
    return combined_sim

def create_interface(args):
    root = tk.Tk()
    root.title("YIYUAN打卡系統")
    root.geometry("350x350")

    # 加載圖片並顯示，假設圖片文件名為"face_recognition_icon.png"，位於與此腳本相同的目錄中
    img = Image.open("C:/Users/user/Downloads/logo.jpg")
    img = img.resize((200, 200), Image.Resampling.LANCZOS)  # 調整圖片大小
    photo = ImageTk.PhotoImage(img)
    label_img = tk.Label(root, image=photo)
    label_img.pack(expand=True)  # 使用expand來使圖片置中

    # 調整Label和Entry的字體大小
    label_id = tk.Label(root, text="Enter Your Name:", font=('Arial', 14, 'bold'))
    label_id.pack(pady=5)  # 增加pady來增加與上一個控件的間距
    entry_id = tk.Entry(root, font=('Arial', 14, 'bold'))
    entry_id.pack(pady=5)  # 調整間距

    def on_enter_or_button_click():
        run_face_recognition(args, entry_id.get())

    button_run_recognition = tk.Button(root, text="Punch", font=('Arial', 14, 'bold'), command=on_enter_or_button_click)
    button_run_recognition.pack(pady=10)

    # 綁定Enter鍵到on_enter_or_button_click函數
    root.bind('<Return>', lambda event=None: on_enter_or_button_click())

    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='人臉辨識系統')
    parser.add_argument('--face_csv_path', type=str, default='C:/Users/user/Desktop/shuo/face_id/face_data.csv', help='人臉資料csv路徑')
    parser.add_argument('--log_file_path', type=str, default='C:/Users/user/Desktop/shuo/face_id/punch_data/', help='打卡csv路徑')#'C:/Users/user/Desktop/shuo/face_id/punch_in_data.csv'
    
    args = parser.parse_args()
    create_interface(args)
