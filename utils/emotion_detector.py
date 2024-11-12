import cv2
from deepface import DeepFace  # type: ignore

def detect_emotion_in_video(video_path):
    try:
        # Load video
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"Error opening video file {video_path}")
            return None
        
        # Initialize a variable to store emotions
        emotions = []

        # Loop through the video frame by frame
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Detect emotions in the current frame using DeepFace
            try:
                # DeepFace returns a list of dictionaries for emotions detected in the frame
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # Extract the dominant emotion from the analysis results
                dominant_emotion = analysis[0]['dominant_emotion']
                emotions.append(dominant_emotion)
            except Exception as e:
                print(f"Error analyzing frame: {e}")
                continue
        
        video.release()  # Release the video capture object

        if emotions:
            # Return the most common emotion detected in the video
            return max(set(emotions), key=emotions.count)
        else:
            return None

    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None
