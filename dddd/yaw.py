import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angles(landmarks):

    # จุด Landmark for cal
    nose_tip = np.array([landmarks[4][0], landmarks[4][1], landmarks[4][2]])       # ปลายจมูก
    nose_bridge = np.array([landmarks[6][0], landmarks[6][1], landmarks[6][2]])    # สันจมูก
    left_eye = np.array([landmarks[33][0], landmarks[33][1], landmarks[33][2]])    # ตาซ้าย
    right_eye = np.array([landmarks[263][0], landmarks[263][1], landmarks[263][2]])  # ตาขวา
    chin = np.array([landmarks[152][0], landmarks[152][1], landmarks[152][2]])     # คาง
    
    # คำนวณเวกเตอร์

    # เวกเตอร์แนวนอน (ตาซ้าย -> ตาขวา)
    eye_vector = right_eye - left_eye
    
    # เวกเตอร์แนวดิ่ง (สันจมูก -> ปลายจมูก)
    vertical_vector = nose_tip - nose_bridge
    
    # เวกเตอร์ความลึก (สันจมูก -> คาง)
    depth_vector = chin - nose_bridge
    
    # สร้างเวกเตอร์อ้างอิง
    reference_horizontal = np.array([1, 0, 0])  # แกน x
    reference_vertical = np.array([0, 1, 0])    # แกน y
    reference_depth = np.array([0, 0, 1])       # แกน z
    
    # คำนวณมุม
    # Yaw (หันซ้าย-ขวา): มุมระหว่างเวกเตอร์ตาและเวกเตอร์อ้างอิงแนวนอน
    yaw = np.arccos(np.dot(eye_vector[:2], reference_horizontal[:2]) / 
                   (np.linalg.norm(eye_vector[:2]) * np.linalg.norm(reference_horizontal[:2])))
    
    # ตรวจสอบทิศทางของ yaw (หันซ้ายหรือขวา)
    if eye_vector[1] < 0:
        yaw = -yaw
    
    # Pitch (ก้ม-เงย): มุมระหว่างเวกเตอร์แนวดิ่งและเวกเตอร์อ้างอิงแนวดิ่ง
    pitch = np.arccos(np.dot(vertical_vector[:2], reference_vertical[:2]) / 
                     (np.linalg.norm(vertical_vector[:2]) * np.linalg.norm(reference_vertical[:2])))
    
    # ตรวจสอบทิศทางของ pitch (ก้มหรือเงย)
    if vertical_vector[0] < 0:
        pitch = -pitch
        
    # Roll (เอียงซ้าย-ขวา): มุมระหว่างเวกเตอร์ตาและแนวนอน
    roll_vector = right_eye - left_eye
    roll = np.arctan2(roll_vector[1], roll_vector[0])
    
    # แปลงมุมจาก radian เป็นองศา
    pitch_deg = pitch * 180 / math.pi
    yaw_deg = yaw * 180 / math.pi
    roll_deg = roll * 180 / math.pi
    
    return pitch_deg, yaw_deg, roll_deg

# set MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


image_path = "/Users/plyfah/Desktop/angry.jpg"  
image = cv2.imread(image_path)

# function check path photo
if image is None:
    print(f"can not load photo {image_path}")
else:
    print("load succes")
    
    # แปลงสี BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image_rgb)
    
    # ถ้าจับหน้าได้
    if results.multi_face_landmarks:
        print("พบใบหน้าในรูปภาพ")
        for face_landmarks in results.multi_face_landmarks:

            # ดึงจุด landmarks ของใบหน้า
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # คำนวณมุม
            pitch, yaw, roll = calculate_angles(landmarks)
            print(f'Pitch (ก้มเงย): {pitch:.2f}°, Yaw (หันซ้าย-ขวา): {yaw:.2f}°, Roll (เอียง): {roll:.2f}°')
            
            # วาดจุด landmarks ในรูปภาพบนเฟรม
            height, width, _ = image.shape
            for i, landmark in enumerate(landmarks):
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                
                # วาดจุดสำคัญที่ใช้ในการคำนวณด้วยสีแดง
                if i in [4, 6, 33, 263, 152]:
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                else:
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            
            # แสดงค่ามุมบนในเฟรม
            cv2.putText(image, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Roll: {roll:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # แสดงผลภาพ
        cv2.imshow('Face Mesh', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("ไม่พบใบหน้าในรูป")