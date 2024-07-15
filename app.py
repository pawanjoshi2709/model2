import cv2
from deepface import DeepFace
from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import tempfile
import os
from mtcnn import MTCNN

app = Flask(__name__)

# Global variable to store the processing progress
processing_progress = 0

def generate_frames(video_source, output_filename):
    global processing_progress
    cap = cv2.VideoCapture(video_source)
    detector = MTCNN()
    frame_number = 0

    # Get the original video frame size and frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MPEG-4 Part 2 codec
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_interval = int(fps / 3)  # Process 3 frames per second for video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    last_bounding_boxes = []
    last_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number % frame_interval != 0:
            # Draw the last detected bounding boxes and emotions on the frame
            for (x, y, x2, y2), emotion in zip(last_bounding_boxes, last_emotions):
                font_scale = 9 * (width / 1080)
                thickness = int(30 * (width / 1080))
                text = f'Emotion: {emotion}'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            
            # Write frame with previous bounding boxes and text to output video
            out.write(frame)

            # Encode the frame to bytes for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            continue

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)
        last_bounding_boxes = []
        last_emotions = []
        for idx, face in enumerate(faces):
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height

            # Analyze emotions for each face
            face_region = frame[y:y2, x:x2]
            if face_region.size == 0:
                continue
            emotion = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            dominant_emotion = emotion[0]['dominant_emotion']

            # Store the bounding box and emotion
            last_bounding_boxes.append((x, y, x2, y2))
            last_emotions.append(dominant_emotion)

            # Calculate font scale and thickness based on bounding box size
            font_scale = int(0.6 * (width / 350))
            thickness = int(2 * (width / 350))

            # Display the emotion and face information on the frame
            text = f'Emotion: {dominant_emotion}'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

        # Write frame with current bounding boxes and text to output video
        out.write(frame)

        # Update the processing progress
        processing_progress = int((frame_number / total_frames) * 100)

        # Encode the frame to bytes for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()
    processing_progress = 100  # Ensure progress is 100% at the end

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global processing_progress
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Reset processing progress
        processing_progress = 0

        # Save the uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # Output filename for processed video
        output_filename = 'output_' + file.filename
        output_path = os.path.join(os.path.dirname(__file__), output_filename)

        # Generate frames and save the processed video
        for _ in generate_frames(temp_path, output_filename):
            pass

        # Return the path to the output video file
        return render_template('index.html', video_path=output_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(os.path.dirname(__file__), filename, as_attachment=True)

@app.route('/progress')
def progress():
    global processing_progress
    return jsonify(progress=processing_progress)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
