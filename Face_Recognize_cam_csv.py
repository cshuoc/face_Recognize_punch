import os
import torch
import numpy as np

import cv2
import csv
import pandas as pd
import datetime

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
    parser.add_argument('--log_file_path', type=str, default='C:/Users/user/Desktop/shuo/face_id/punch_data/', help='打卡csv路徑')#'C:/Users/user/Desktop/shuo/face_id/punch_in_data.csv'
    
    args = parser.parse_args()
    create_interface(args)
