from ultralytics import YOLO

# Load your model
model = YOLO(r"D:\CODE\DS-Project\Usman Model\best.pt")

# Run detection on an example image
results = model.predict(source=r"D:\CODE\DS-Project\Usman Model\test_image.jpg")

# Print results
print(results)
