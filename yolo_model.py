import os

from ultralytics import YOLOv10

config_path = 'C:\\Users\\user\\OneDrive\\Desktop\\Shoe Detection\\config.yaml'

model =  YOLOv10.from_pretrained("jameslahm/yolov10n")

model.train(data=config_path, epochs=100, batch=32)