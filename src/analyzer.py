import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import av
from streamlit.runtime.uploaded_file_manager import UploadedFile
from deepface import DeepFace
from io import BytesIO

keras.config.disable_interactive_logging()

emotion_labels = list(map(str.lower, ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']))



class Analyzer:
    """ Audience Emotion Analyzer class """
    def analyze(self, 
        file: UploadedFile | None = None, 
        confidence: float | None = .5) -> tuple[bool, pd.DataFrame]:
        
        """ Analyze video in file """
        # Check for file
        if file is None:
            raise ValueError('Must have a file to analyze.')

        # Load video and get properties
        container = av.open(file, mode="r")
        stream = container.streams.video[0]
        # Set video output path
        output_path = "classified_output.mp4"
        # Configure a video writer
        fourcc = cv2.VideoWriter_fourcc(*'.mp4')
        fps = stream.average_rate if stream.average_rate else 24
        out = None
        # Iterate over the frames in the stream
        i = 0
        for i, frame in enumerate(container.decode(stream)):
            # Convert to RGB and rotate
            frame = frame.to_rgb().to_ndarray()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # Try to classify the frame
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
                for face in results:
                    # If the output stream is None, we initialize it
                    if out is None:
                        frame_h, frame_w, _ = frame.shape
                        out = cv2.VideoWriter(output_path, fourcc, float(fps), (frame_w, frame_h))                    
                    # Get the face region and dominant emotion
                    region = face['region']
                    x, y, w, h = int(region['x']), int(region['y']), int(region['w']), int(region['h'])
                    emotion = face['dominant_emotion']
                    emotion_probs = face['emotion']
                    label = emotion.capitalize()
                    dominant_prob = emotion_probs[emotion]
                    # If the probability is larger than the confidence we display a bounding rect
                    if dominant_prob > confidence:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except:
                pass
            # Write the processed frame to the output video
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))    
        # Release the output stream and close the container
        out.release()
        container.close()
        # Get the video bytes so that we can return them
        video_bytes = BytesIO()
        with open(output_path, "rb") as f:
            video_bytes.write(f.read())
        video_bytes.seek(0)

        return video_bytes