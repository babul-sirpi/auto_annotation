import cv2
import shutil
from pathlib import Path
import os
from ultralytics import YOLO
import numpy as np
import torch

def extract_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Check if the video file is successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return []
    else:
        print("Video loaded successfully")
    # Output path
    output_path = "output_path"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Initialize frame count and timestamp variables
    frame_count = 0
    timestamp = 0

    while True:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert to milliseconds

        # Read the next frame
        success, frame = video.read()

        # Break the loop if no frame is read
        if not success:
            break

        # Generate a frame file name
        frame_name = f"{output_path}/frame_{frame_count}.jpg"

        # Save the frame as an image file
        cv2.imwrite(frame_name, frame)
        # Increment the frame count
        frame_count += 1

        # Update the timestamp to the next second
        timestamp += 1

    # Release the video file
    video.release()
    print(f"Extracted {frame_count} frames")
    return os.path.abspath(output_path)
def annotate_frames(output_path):
    def auto_annotate(data, det_model='yolov8x.pt', output_dir=None):
        """
    Automatically annotates images using a YOLO object detection model and saves the coordinates of the detected region.
    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        output_dir (str, None, optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        det_model = YOLO(det_model)
        det_model.to(device)

        if not output_dir:
            output_dir = Path(str(data)).parent / 'labels'
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        det_results = det_model(data, conf = 0.7,verbose=False)

        for result in det_results:
            class_ids = result.boxes.cls.int().tolist()  # Get class IDs
            if len(class_ids):
                with open(str(Path(output_dir) / Path(result.path).stem) + '.txt', 'w') as f:
                    for i, class_id in enumerate(class_ids):
                        bbox = result.boxes.xyxyn[i].tolist() # Access each bounding box using the loop index
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3] # Unpack the coordinates
                        
                        # Calculate x_center, y_center, width, height
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')  # Save normalized coordinates of the detected region
        
    model_path= "best.pt"
    data = output_path
    output_dir = "output_label_dir"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    auto_annotate(data=data,det_model=model_path,output_dir=output_dir)
    return os.path.abspath(output_dir)

def image_annota(output_path,output_dir):
    new_annotation = "new_annotation_path"
    if not os.path.exists(new_annotation):
        os.makedirs(new_annotation)
    labels_dir = output_dir
    extracted_frames_path = output_path
    count=0
    for i in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, i)
        destination_path = os.path.join(new_annotation, i)
        
        if os.path.exists(destination_path):
            # If the file already exists, make a copy
            destination_copy_path = os.path.join(new_annotation, f"copy_{count}" + i)
            shutil.copy2(label_path, destination_copy_path)
            print(f"File {i} already exists. Created a copy: {destination_copy_path}")
            os.remove(label_path) # removing after copying
            count+=1
        else:
            # If the file doesn't exist, move it to the destination
            shutil.move(label_path, destination_path)
            print(f"Moved {i} to {destination_path}")
    count=0
    for i in os.listdir(extracted_frames_path):
        image_path = os.path.join(extracted_frames_path, i)
        
        destination_path = os.path.join(new_annotation, i)
        if os.path.exists(destination_path):
            # If the file already exists, make a copy
            destination_copy_path = os.path.join(new_annotation, f"copy_{count}" + i)
            shutil.copy2(image_path, destination_copy_path)
            os.remove(image_path) # removing after copying
            count+=1
        else:
            # If the file doesn't exist, move it to the destination
            shutil.move(image_path, new_annotation)
            print(f"Moved {i} to {destination_path}")
    return os.path.abspath(new_annotation)
    
def delete_image_files_without_text(annotation_path):
    image_directory = annotation_path
    text_dir = annotation_path
    for i in os.listdir(image_directory):
        file,ext = os.path.splitext(i)
        if ext==".jpg":
            text_file_path= os.path.join(text_dir,f"{file}.txt")
            if not os.path.exists(text_file_path):
                os.remove(os.path.join(image_directory,i))
    return annotation_path
def data_prep(video_path):
    output_path = extract_frames(video_path)
    output_dir_path = annotate_frames(output_path)
    annotation_dir = image_annota(output_path,output_dir_path)
    annotation_path = delete_image_files_without_text(annotation_dir)
    print("Frames_extraction and annotation completed")

data_prep(r"/home/babul/babul/yolov8/video_20230310_102105 - Thasthageer Dawood A R.mp4")
    