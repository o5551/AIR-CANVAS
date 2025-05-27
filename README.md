# 🖌️ Air Canvas using Hand Gestures

An interactive virtual drawing application that allows users to draw in the air using their finger as a pen — no mouse or stylus required. Built with **Python**, **OpenCV**, and **MediaPipe**, the system captures hand gestures through webcam input to draw, change colors, and clear the canvas in real time.

---

## 🧠 Features

- ✋ Real-time hand tracking using **MediaPipe**
- 🎨 Draw on a virtual canvas using **index finger**
- 🌈 Gesture-based **color switching** (Blue, Green, Red, Yellow)
- 🧽 Gesture-based **clear canvas** option
- 📐 Basic shape-drawing functions prepared (rectangle, circle, square)
- 🖥️ Intuitive UI with OpenCV canvas and live webcam feed

---

## 📦 Tech Stack

- **Python 3.6+**
- **OpenCV** for image processing and GUI
- **MediaPipe** for hand landmark detection
- **NumPy** for geometric calculations
- **collections.deque** for stroke history management

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/air-canvas.git
cd air-canvas
```
2. Install Required Packages
Make sure Python and pip are installed, then run:
```
pip install opencv-python mediapipe numpy
```
3. Run the Script
   ```
   python air_canvas.py
   ```
🧠 How It Works
The webcam captures live video and flips it for a mirror-like view.

MediaPipe detects hand landmarks (21 points), especially the index finger tip and thumb.

If only the index finger is up, it starts drawing.

If the hand hovers over color boxes on the top bar, it switches drawing colors.

If the hand moves to the “CLEAR” box, it resets the canvas.

📁 File Structure
```
📦 air-canvas
 ┣ 📄 air_canvas.py         # Main application code
 ┣ 📄 README.md             # Project documentation
```
🙌 Contributing
Contributions are welcome!
Feel free to fork this repo, suggest features, or submit pull requests.

📬 Contact
For feedback or questions, feel free to reach out via GitHub Issues or email.

---

✅ Let me know if you'd like to:
- Add a **GIF/screenshot** of the tool in action
- Include a **Makefile**
- Set up this for **cross-platform** use or packaging

I'm happy to help further polish your repo!









