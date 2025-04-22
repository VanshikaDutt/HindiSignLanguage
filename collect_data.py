import os
import cv2

# Use transliterated folder names to avoid Unicode path issues
vowels = ['a', 'aa', 'i', 'ii', 'u', 'uu', 'ri', 'e', 'ai', 'o', 'au']
directory = os.path.abspath("Image")

# Key mapping stays the same
key_mapping = {
    ord('1'): vowels[0],
    ord('2'): vowels[1],
    ord('3'): vowels[2],
    ord('4'): vowels[3],
    ord('5'): vowels[4],
    ord('6'): vowels[5],
    ord('7'): vowels[6],
    ord('8'): vowels[7],
    ord('9'): vowels[8],
    ord('0'): vowels[9],
    ord('-'): vowels[10],  # 'au'
}

# Create folders
for vowel in vowels:
    folder = os.path.join(directory, vowel)
    os.makedirs(folder, exist_ok=True)

print(f"📂 Saving images to: {directory}")
print("📸 Press keys 1-0 or '-' to save images. Press ESC to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    x, y, w, h = 100, 100, 300, 300
    roi = frame[y:y + h, x:x + w]

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key in key_mapping:
        vowel = key_mapping[key]
        folder = os.path.join(directory, vowel)
        count = len(os.listdir(folder))
        filename = os.path.join(folder, f"{count}.png")

        print(f"📝 Saving: {filename}")
        if cv2.imwrite(filename, roi):
            print(f"✅ Saved: {filename}")
        else:
            print(f"❌ Failed to save: {filename}")

cap.release()
cv2.destroyAllWindows()