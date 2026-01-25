import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera Benchmark started. Press 'q' to quit.")
    
    # Initialize variables for FPS calculation
    prev_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read frame.")
            break
            
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Overlay FPS on the frame
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow("Camera Benchmark", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
