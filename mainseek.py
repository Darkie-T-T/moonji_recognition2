import tkinter as tk
from tkinter import ttk, messagebox
import cv2, os, csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

#================= FUNCTIONS ========================
def assure_path_exists(path):
    """Ensure a directory exists; if not, create it."""
    os.makedirs(path, exist_ok=True)

def tick():
    """Update the clock every 200ms."""
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

def contact():
    """Display contact information."""
    messagebox.showinfo("Contact", "Contact us at: contact@attendance.com")

def check_haarcascadefile():
    """Check if the Haar Cascade file exists."""
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        messagebox.showerror("Error", "Missing haarcascade file!")
        window.destroy()

#================= IMAGE CAPTURE =====================
def TakeImages():
    """Capture images for a new student."""
    check_haarcascadefile()
    Id = txt_id.get()
    name = txt_name.get()
    
    if not Id or not name:
        messagebox.showerror("Error", "Both ID and Name are required!")
        return
    
    if not name.replace(' ', '').isalpha():
        messagebox.showerror("Error", "Name must contain only alphabets!")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sampleNum = 0
    img_count = 0
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            sampleNum += 1
            cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
            img_count = sampleNum
            
        cv2.imshow('Capturing Faces', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
            break
            
    cam.release()
    cv2.destroyAllWindows()
    
    # Save student details
    assure_path_exists("StudentDetails")
    with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([Id, name])
    
    messagebox.showinfo("Success", f"Captured {img_count} images for {name}")

#================= TRAINING ==========================
def TrainImages():
    """Train the face recognition model."""
    check_haarcascadefile()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    faces, ids = [], []
    for image_path in [os.path.join("TrainingImage", f) for f in os.listdir("TrainingImage")]:
        pil_image = Image.open(image_path).convert('L')
        img_np = np.array(pil_image, 'uint8')
        Id = int(os.path.split(image_path)[-1].split(".")[1])
        
        faces.append(img_np)
        ids.append(Id)
        
    recognizer.train(faces, np.array(ids))
    assure_path_exists("TrainingImageLabel")
    recognizer.save("TrainingImageLabel/Trainner.yml")
    messagebox.showinfo("Success", "Training completed successfully!")

#================= ATTENDANCE TRACKING ================
def TrackImages():
    """Track attendance using the trained model."""
    check_haarcascadefile()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    except FileNotFoundError:
        messagebox.showerror("Error", "Student database not found!")
        return

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance_log = []
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            
            if conf < 60:
                student = df[df['Id'] == Id]
                name = student['Name'].values[0]
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_log.append([Id, name, ts])
                cv2.putText(img, name, (x, y+h+20), font, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y+h+20), font, 0.8, (0, 0, 255), 2)
                
        cv2.imshow('Attendance System', img)
        if cv2.waitKey(1) == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()
    
    # Save attendance
    if attendance_log:
        date = datetime.datetime.now().strftime("%Y%m%d")
        assure_path_exists("Attendance")  # Ensure directory exists
        file_path = f"Attendance/Attendance_{date}.csv"
        
        try:
            # Check if file exists and is not empty
            file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
            
            with open(file_path, "a+", newline='') as f:
                writer = csv.writer(f)
                # Write header if file is empty or doesn't exist
                if not file_exists:
                    writer.writerow(["ID", "Name", "Timestamp"])
                writer.writerows(attendance_log)
            
            messagebox.showinfo("Success", "Attendance recorded successfully!")
            UpdateTable()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save attendance: {e}")
    else:
        messagebox.showinfo("Info", "No attendance records to save")

#================= UPDATE TABLE ======================
def UpdateTable():
    """Update the attendance table in the GUI."""
    date = datetime.datetime.now().strftime("%Y%m%d")
    try:
        df = pd.read_csv(f"Attendance/Attendance_{date}.csv")
        tv.delete(*tv.get_children())
        for index, row in df.iterrows():
            tv.insert("", "end", values=(row["ID"], row["Name"], row["Timestamp"]))
    except FileNotFoundError:
        messagebox.showinfo("Info", "No attendance records found for today")

#================= GUI SETUP ==========================
window = tk.Tk()
window.title("AI Attendance System")
window.geometry("1280x720")
window.configure(bg="#2c3e50")

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", background="#34495e", fieldbackground="#2c3e50", foreground="white")

# Header
header = tk.Frame(window, bg="#34495e")
header.pack(fill="x", padx=10, pady=10)
tk.Label(header, text="AI Powered Attendance System", font=("Helvetica", 24), 
        bg="#34495e", fg="white").pack(pady=10)

# Main Content
main_frame = tk.Frame(window, bg="#2c3e50")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Registration Section
reg_frame = tk.LabelFrame(main_frame, text="Student Registration", font=("Helvetica", 14),
                         bg="#34495e", fg="white")
reg_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

tk.Label(reg_frame, text="Student ID:", bg="#34495e", fg="white").grid(row=0, column=0, padx=5, pady=5)
txt_id = tk.Entry(reg_frame, width=25, font=("Helvetica", 12))
txt_id.grid(row=0, column=1, padx=5, pady=5)

tk.Label(reg_frame, text="Student Name:", bg="#34495e", fg="white").grid(row=1, column=0, padx=5, pady=5)
txt_name = tk.Entry(reg_frame, width=25, font=("Helvetica", 12))
txt_name.grid(row=1, column=1, padx=5, pady=5)

btn_frame = tk.Frame(reg_frame, bg="#34495e")
btn_frame.grid(row=2, columnspan=2, pady=10)

tk.Button(btn_frame, text="Capture Images", command=TakeImages, width=15,
         bg="#3498db", fg="white").pack(side="left", padx=5)
tk.Button(btn_frame, text="Train Model", command=TrainImages, width=15,
         bg="#2ecc71", fg="white").pack(side="left", padx=5)

# Attendance Section
att_frame = tk.LabelFrame(main_frame, text="Attendance Records", font=("Helvetica", 14),
                         bg="#34495e", fg="white")
att_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

tv = ttk.Treeview(att_frame, columns=("ID", "Name", "Time"), show="headings", height=10)
tv.heading("ID", text="ID")
tv.heading("Name", text="Name")
tv.heading("Time", text="Time")
tv.pack(fill="both", expand=True, padx=5, pady=5)

scroll = ttk.Scrollbar(att_frame, orient="vertical", command=tv.yview)
scroll.pack(side="right", fill="y")
tv.configure(yscrollcommand=scroll.set)

# Control Panel
ctrl_frame = tk.Frame(main_frame, bg="#2c3e50")
ctrl_frame.grid(row=1, columnspan=2, pady=10)

tk.Button(ctrl_frame, text="Take Attendance", command=TrackImages, width=20,
         bg="#e67e22", fg="white").pack(side="left", padx=10)
tk.Button(ctrl_frame, text="Exit", command=window.destroy, width=20,
         bg="#e74c3c", fg="white").pack(side="left", padx=10)

# Status Bar
status_bar = tk.Frame(window, bg="#34495e", height=30)
status_bar.pack(fill="x", side="bottom")
clock = tk.Label(status_bar, bg="#34495e", fg="white")
clock.pack(side="right", padx=10)

# Initialize
assure_path_exists("TrainingImage")
assure_path_exists("StudentDetails")
assure_path_exists("Attendance")
tick()

window.mainloop()
