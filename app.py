from flask import Flask, request, render_template, redirect, url_for


from werkzeug.utils import secure_filename
import os
import cv2
from deepface import DeepFace




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')


uploaded_videos = []

@app.route('/')
def home():
    return render_template('upload.html', videos=uploaded_videos)


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is part of the request
    file = request.files.get('file')
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print(f"Received file: {filename}")
        print(f"File path to save: {file_path}")
        
        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        try:
            # Save the file
            file.save(file_path)
            print(f"File saved successfully: {file_path}")

            # Detect emotion from the video
            emotion = detect_emotion_from_video(file_path)
            print(f"Emotion detected: {emotion}")

            # Add the uploaded video to the list
            uploaded_videos.append({'filename': filename, 'emotion': emotion})

            # Return the success page
            return render_template('upload_success.html', video_filename=filename, emotion=emotion)
        
        except Exception as e:
            print(f"Error saving file or detecting emotion: {e}")
            return render_template('upload.html', message="Error processing video. Please try again.")
    
    return render_template('upload.html', message="Please upload a video file.")

def detect_emotion_from_video(video_path):
    # Load the video using OpenCV
    video = cv2.VideoCapture(video_path)

    # Check if video is opened correctly
    if not video.isOpened():
        return "Error loading video"

    frame_count = 0
    emotion_results = []
    frame_interval = 10  # Process every 10th frame

    while True:
        ret, frame = video.read()
        if not ret:
            break  # Break if no more frames

        if frame_count % frame_interval == 0:
            # Convert frame to RGB and analyze emotion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_results.append(dominant_emotion)
            except Exception as e:
                print(f"Error in emotion detection: {e}")

        frame_count += 1

    video.release()

    # If there were no valid frames, return a default emotion
    if frame_count == 0:
        return "No faces detected"

    # Return the most frequent emotion from the video
    most_frequent_emotion = max(set(emotion_results), key=emotion_results.count)
    return most_frequent_emotion

@app.route('/delete/<filename>', methods=['POST'])
def delete_video(filename):
    # Find the video file in the list
    video = next((video for video in uploaded_videos if video['filename'] == filename), None)

    if video:
        # Delete the video file from the filesystem
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted video: {filename}")

        # Remove the video from the uploaded videos list
        uploaded_videos.remove(video)

    return redirect(url_for('home'))





if __name__ == "__main__":
    app.run(debug=True)