import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

# 調用視訊鏡頭
cap = cv2.VideoCapture(0)

while(True):
    # 擷取攝像頭拍攝到的畫面
    ret, frame = cap.read()
    if ret is not True:
        print('沒有偵測到鏡頭')
        break
    faces = face_cascade.detectMultiScale(frame, 1.3, 2)
    img = frame
    for (x,y,w,h) in faces:
    	# 藍色框畫出人臉
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    	# 框選出人臉區域，在人臉區域而不是全圖中進行人眼檢測，節省計算資源
        face_area = img[y:y+h, x:x+w]
        
        ## 人眼檢測
        # 用人眼級聯分類器引擎在人臉區域進行人眼識別，返回的eyes為眼睛坐標列表
        eyes = eye_cascade.detectMultiScale(face_area,1.3,10)
        for (ex,ey,ew,eh) in eyes:
            # 綠色框畫出人眼
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        
        ## 微笑檢測
        # 用微笑級聯分類器引擎在人臉區域進行識別，返回的smiles為坐標列表
        smiles = smile_cascade.detectMultiScale(face_area,scaleFactor= 1.16,minNeighbors=65,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex,ey,ew,eh) in smiles:
            # 紅色框畫出微笑
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
            cv2.putText(img,'微笑',(x,y-7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        
	# 畫面
    cv2.imshow('frame2',img)
    # 每5毫秒回傳鍵盤動作
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 關閉所有窗口
cap.release()
cv2.destroyAllWindows()
