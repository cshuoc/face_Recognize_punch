from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import argparse

def recognize_face(image_path, mtcnn, model):
    # 加載圖像
    img = Image.open(image_path)
    img = img.resize((256, 256))
    # 檢測圖像中的所有人臉
    boxes, _ = mtcnn.detect(img)
    
    # 畫出檢測到的人臉
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    print('boxes : ',boxes)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # 使用MTCNN提取人臉
    faces = mtcnn(img)
    
    # 使用InceptionResnetV1計算所有檢測到的人臉的嵌入向量
    embeddings = model(faces)
    img_draw.show()
    return embeddings

def similarity(embedding1, embedding2, alpha=0.2, max_dist=10.0):
    # 將embedding1和embedding2轉為一維向量
    embedding1 = embedding1.view(1, -1)
    embedding2 = embedding2.view(1, -1)

    # 餘弦相似度
    cosine_sim = F.cosine_similarity(embedding1, embedding2).item()
    
    # 歐式距離且轉為相似度
    euclidean_dist = torch.norm(embedding1 - embedding2).item()
    euclidean_sim = 1 - (euclidean_dist / max_dist)
    
    # 加權
    combined_sim = alpha * cosine_sim + (1 - alpha) * euclidean_sim
    print(combined_sim)
    return combined_sim


def run(image_path1, image_path2, mtcnn, model, threshold = 0.85):
    embeddings1 = recognize_face(image_path1, mtcnn, model)
    embeddings2 = recognize_face(image_path2, mtcnn, model)
    sim = similarity(embeddings1, embeddings2)
    print('閾值   : ', threshold)
    print('相似度 : ', sim)
    print('結果   : 匹配!!!') if sim > threshold else print('結果 : 不匹配...')

if __name__ == '__main__':
    # 初始化命令行參數解析器
    parser = argparse.ArgumentParser(description='人臉識別')
    # 添加圖像路徑參數
    parser.add_argument('--img1_path', type=str, help='第一張圖像的路徑')
    parser.add_argument('--img2_path', type=str, help='第二張圖像的路徑')
    parser.add_argument('--threshold', type=float, default=0.88, help='相似度閾值，默認為0.85')

    # 解析命令行參數
    args = parser.parse_args()

    mtcnn = MTCNN(keep_all=True)
    model = InceptionResnetV1(pretrained='vggface2').eval()

    # 使用從命令行參數獲取的圖像路徑和閾值
    run(args.img1_path, args.img2_path, mtcnn, model, args.threshold)
