import cv2 as cv

# open cv net (weights and configuration files)
net = cv.dnn.readNet("yolo/yolov4-tiny.weights","yolo/yolov4-tiny.cfg")


# opencv model
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

# read objects 
objects = []
with open("yolo/classes.txt","r") as file:
    objects = file.read().split("\n")
    
# initialize camera
capture = cv.VideoCapture(0)

# set camera resolution
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


# for video capturing
while True:
    # get frames: video için toplanan fotoğraf parçaları
    ret, frame = capture.read()
    
    # object detection
    # objectIds: - confs: confidence(tutarlılık) - bbox: bounding box
    objectIds, confs, bbox = model.detect(frame)
    
    # zip for multiple iterations same time
    for objectId, conf, box in zip(objectIds, confs, bbox):
        x, y, width, height = box
        
        cv.putText(frame, objects[objectId], (x, y-10), cv.FONT_HERSHEY_PLAIN, 1, (63,0,113), 2)
        cv.putText(frame, str(conf*100), (x+150,y), cv.FONT_HERSHEY_PLAIN, 1, (63,0,113), 2)
        cv.rectangle(frame, (x,y), (x+width, y+height), (63,0,113), 3,)
        
        
    # display
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF==ord("q"):
        break
    
capture.release()
cv.destroyAllWindows()  

  