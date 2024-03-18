
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import csv
import numpy as np
from PIL import Image

# 載入 MTCNN 和 InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

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

def add_new_face(image_path, label, mtcnn, model, face_database):
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        embedding = model(img_cropped).detach().cpu().squeeze()
        face_database[label] = {'embedding': embedding, 'label': label}
        save_face_database(face_database, csv_file_path)  # 更新 CSV

def recognize_face(image_path, mtcnn, model, face_database, threshold=0.85):
    
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        embedding = model(img_cropped).detach().cpu().squeeze()

        highest_sim = 0
        min_label = None
        for label, data in face_database.items():
            known_embedding = data['embedding']
            sim = similarity(embedding, known_embedding)
            if sim > highest_sim:
                highest_sim = sim
                min_label = label

        if highest_sim > threshold:
            print(f"你是{face_database[min_label]['label']}，相似度 : {highest_sim}")
        else:
            print("我不認識你/妳")

def similarity(embedding1, embedding2):
    embedding1 = embedding1.view(1, -1)
    embedding2 = embedding2.view(1, -1)
    cosine_sim = F.cosine_similarity(embedding1, embedding2).item()
    euclidean_dist = torch.norm(embedding1 - embedding2).item()
    euclidean_sim = 1 - euclidean_dist / 10.0
    combined_sim = 0.5 * cosine_sim + 0.5 * euclidean_sim
    return combined_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人臉辨識系統')

    parser.add_argument('--img_path', type=str, default='', help='欲辨識圖片的路徑')
    parser.add_argument('--threshold', type=float, default=0.8, help='相似度閥值')
    parser.add_argument('--add_new_face', type=bool, default=False, help='新增面孔')
    parser.add_argument('--new_face_name', type=str, default='', help='新面孔圖片的名字')
    parser.add_argument('--new_face_path', type=str, default='', help='新面孔圖片的路徑')
    parser.add_argument('--face_csv_path', type=str, default='', help='記錄人臉資料csv路徑')
    args = parser.parse_args()

    csv_file_path = args.face_csv_path

    # 載入人臉資料庫
    face_database = load_face_database(csv_file_path)

    # 添加新的人臉數據到資料庫
    if args.add_new_face is True:
        face_paths_labels = [
            #('PICTURE_1', 'NAME_1'),
            # ('PICTURE_2', 'NAME_2'),
            # ('PICTURE_3', 'NAME_3'),
            (args.new_face_path, args.new_face_name),
        ]
        for path, label in face_paths_labels:
            add_new_face(path, label, mtcnn, model, face_database)

    # 辨識圖片中的人臉
    image_paths = [
        # 'img_path_1',
        # 'img_path_2',
        # 'img_path_3',
        args.img_path,
    ]
    for image_path in image_paths:
        recognize_face(image_path, mtcnn, model, face_database, threshold = args.threshold)
