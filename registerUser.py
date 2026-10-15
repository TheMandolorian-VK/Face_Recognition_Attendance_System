import cv2
import time
import os

# Initialize the webcam
camera = cv2.VideoCapture(0)

def captureBurst(camera, name):
    print("\n--- Starting burst capture of 5 photos ---")
    for i in range(5):
        ret, frame = camera.read()
        if ret:
            filename = os.path.join("known_faces", name, f"captured_photo{i+1}.png")
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(filename, frame)
            print(f"Photo successfully saved as {filename}")
            # Add a 0.3-second delay so frames are actually different
            time.sleep(0.3) 
        else:
            print("Error: Failed to capture image frame.")
    print("--- Burst complete. Standing by... ---\n")

if not camera.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam active. Controls:")
    print("  - Press SPACEBAR to capture a 5-photo burst")
    print("  - Press ESC to exit safely")
    name = input("Enter full name to register under:")

    while True:
        # Read a frame just to keep the live preview active
        ret, frame = camera.read()
        if not ret:
            print("Error: Camera feed lost.")
            break

        # Display the live video feed in a window
        cv2.imshow("Webcam Live Feed", frame)

        # Wait 1 millisecond for a key press
        key = cv2.waitKey(1) & 0xFF

        # If Spacebar (ASCII 32) is pressed, trigger the burst function
        if key == 32:
            captureBurst(camera, name)

        # If ESC (ASCII 27) is pressed, break out of the infinite loop
        elif key == 27:
            print("Exiting program...")
            break

    # Clean up and release the camera safely
    camera.release()
    cv2.destroyAllWindows()
