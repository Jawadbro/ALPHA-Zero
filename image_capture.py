import cv2
import time

def capture_frames_every_n_seconds(video_source=0, interval=2):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    image_count = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            image_count += 1
            
            image_filename = f"image_{image_count}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"Saved: {image_filename}")

        # Display the resulting frame (optional)
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing frames
capture_frames_every_n_seconds()