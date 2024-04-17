from ultralytics import YOLO

model= YOLO('model\yolo5_last.pt')

result= model.predict('videos\input_video.mp4',conf=0.2, save=True)
print(result)