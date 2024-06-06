from ultralytics import YOLO

model=YOLO('models\\best.pt',stream=True)

results=model.predict('input_videos/ref.mp4',save=True)
print(results[0])
print('--------------------------------------------------')
for box in results[0].boxes:
    print(box)