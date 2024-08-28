'''
Alphabets Recoginition
多個英文字母光學辨識
作者: 國立中央大學數學系計算與資料科學組 張軒豪
'''
from tabnanny import verbose
import cv2
from keras.models import load_model
import numpy as np

tracker_list = []
tracker_num = int(input())
for i in range(tracker_num):
    tracker = cv2.TrackerCSRT_create()        # 創建三組追蹤器
    tracker_list.append(tracker)

tracking = False                    # 設定 False 表示尚未開始追蹤

colors = [(255,0,0),(0,255,0),(0,0,255)]

cap = cv2.VideoCapture(0)
print('loading...')
cnn = load_model('emnist_bymerge_cnn_model.xml')   # 載入模型
print('start...')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    frame = cv2.resize(frame,(540,300))  # 縮小尺寸，加快速度
    keyName = cv2.waitKey(1)

    if keyName == ord('q'):
        break
    if keyName == ord('f'):
        tracking = False
    if keyName == ord('a'):   
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #轉灰階
        edged = cv2.Canny(gray_image, 20,160) #Perform Edge detection
        (contours,_)=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #找出所有邊界，RETR_EXTERNAL是取圖型外邊界
        
        contours_list = sorted(contours,key=cv2.contourArea, reverse = True)
        
        for cnt in range(tracker_num):
            area = cv2.boundingRect(contours_list[cnt]) # 用最小矩形框住目標
            tracker_list[cnt].init(frame, area)    # 初始化追蹤器
        tracking = True              # 設定可以開始追蹤
    if tracking:
        for i in range(tracker_num):
            success, point = tracker_list[i].update(frame)   # 追蹤成功後，不斷回傳左上和右下的座標
            if success:
                x, y, w, h = (int(i) for i in point)
                img_num = frame.copy()                     # 複製一個影像作為辨識使用
                img_num = img_num[y:y+h, x:x+w]          # 擷取辨識的區域

                img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
                # 針對白色文字，做二值化黑白轉換，轉成黑底白字
                ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)
                #output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)     # 顏色轉成彩色
                #frame[0:h, 480:480+w] = output                            # 將轉換後的影像顯示在畫面右上角

                img_num = cv2.resize(img_num,(28,28))   # 縮小成 28x28，和訓練模型對照
                img_num = img_num.astype(np.float32)    # 轉換格式
                img_num = img_num.reshape(1,28,28)
                img_num = img_num/255
                img_pre = cnn.predict(img_num, verbose = 0)          # 進行辨識
                #print(img_num.shape)
                num = np.argmax(img_pre)        # 取得辨識結果
                #num = str(num) if num < 10 else (chr(num + 55) if num < 36 else chr(num + 59)) #byclass label 排列
                bymerge = ['a','b','d','e','f','g','h','n','q','r','t']
                num = str(num) if num < 10 else (chr(num + 55) if num < 36 else bymerge[num - 36])

                text = num                              # 印出的文字內容
                org = (x,y-20)                          # 印出的文字位置
                fontFace = cv2.FONT_HERSHEY_SIMPLEX     # 印出的文字字體
                fontScale = 2                           # 印出的文字大小
                color = colors[i%3]                       # 印出的文字顏色
                thickness = 2                           # 印出的文字邊框粗細
                lineType = cv2.LINE_AA                  # 印出的文字邊框樣式
                cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, lineType) # 印出文字

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)  # 標記辨識的區域

    cv2.imshow('', frame)

cap.release()
cv2.destroyAllWindows()