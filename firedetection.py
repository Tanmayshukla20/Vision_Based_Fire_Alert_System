import cv2
import pygame
from twilio.rest import Client
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# Twilio account credentials
from twilio.rest import Client

account_sid = 'AC38b15e4b20d8868a51dac50e04d7fdd2'
auth_token = '46bdf3505b98cdd0b08da25effeb264d'


TWILIO_WHATSAPP_NUMBER = 'whatsapp:‪+14155238886‬'
TARGET_WHATSAPP_NUMBER = 'whatsapp:‪+917999878432‬'

# Initialize twilio account
client = Client(account_sid, auth_token)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load the YOLOv8 model
model = YOLO("best_train(2).pt")


def send_whatsapp_message(message):
    try:
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=TARGET_WHATSAPP_NUMBER
        )
        print("WhatsApp message sent successfully.")
    except Exception as e:
        print("Failed to send WhatsApp message:", e)

        
def trigger_alarm():
    pygame.mixer.init()
    alarm_sound = pygame.mixer.Sound("south-korea-eas-alarm-1966-422162.mp3")
    alarm_sound.play()
    pygame.time.delay(5000)
    pygame.mixer.quit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]

    fire_detected = False

    for box in result.boxes:
        label_index = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = model.model.names[label_index]  # Access correct label names

        if label == 'fire' and confidence >= 0.3:
            
            send_whatsapp_message("Alert Hello from Twilio Sandbox! Fire is detected") 
            model(frame, save=True)  # Save image to runs/detect/predict  1.213
            trigger_alarm()
            fire_detected = True
            break

    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
