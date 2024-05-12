import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

def box_to_coord(box):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    xA, yA, xB, yB = cords
    return xA, yA, xB, yB
def main():
    st.title("Object Detection App")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load YOLO model
        model = YOLO("yolov5s.pt")

        # Read image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(image)
        # Draw bounding boxes on the image
        frame = np.array(image)
        for result in results:
            for box in result.boxes:
                xA, yA, xB, yB = box_to_coord(box)
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # Display the image with bounding boxes
        st.image(frame, channels="BGR")


if __name__ == "__main__":
    main()
