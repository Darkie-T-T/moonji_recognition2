AI-Powered Attendance System Documentation
Overview
This is an AI-powered attendance system that uses facial recognition to mark attendance. It is built using Python and leverages the following technologies:

OpenCV for face detection and image processing

Tkinter for the graphical user interface (GUI)

Pandas for handling CSV data

Pillow for image processing

The system allows administrators to:

Register new students by capturing their facial images

Train a facial recognition model

Take attendance using the trained model

View attendance records in real-time

Features
Student Registration:

Capture multiple facial images for each student

Store student details (ID and Name) in a CSV file

Model Training:

Train a facial recognition model using captured images

Save the trained model for future use

Attendance Tracking:

Recognize registered students in real-time using a webcam

Log attendance with timestamps

Save attendance records in CSV files

User Interface:

Modern and intuitive GUI

Real-time clock display

Clear error messages and feedback

Requirements
Software
Python 3.8 or higher

Required Python packages:

bash
Copy
pip install opencv-python pandas pillow
Hardware
Webcam for capturing images and taking attendance

haarcascade_frontalface_default.xml file (included in OpenCV)

File Structure
Copy
AI-Attendance-System/
├── main.py                  # Main application code
├── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
├── TrainingImage/           # Folder to store captured face images
├── StudentDetails/          # Folder to store student details (CSV)
├── Attendance/              # Folder to store attendance records (CSV)
├── TrainingImageLabel/      # Folder to store the trained model
How It Works
Student Registration:

Enter the student's ID and Name in the registration section.

Click "Capture Images" to take multiple facial images (up to 100).

Images are saved in the TrainingImage folder.

Model Training:

Click "Train Model" to train the facial recognition model.

The trained model is saved as Trainner.yml in the TrainingImageLabel folder.

Attendance Tracking:

Click "Take Attendance" to start the webcam.

The system will recognize registered students and log their attendance.

Attendance records are saved in the Attendance folder with a timestamp.

View Attendance:

Attendance records are displayed in a table within the application.

Code Structure
Key Functions
TakeImages():

Captures facial images for a new student.

Saves images in the TrainingImage folder.

Stores student details in StudentDetails.csv.

TrainImages():

Trains the facial recognition model using captured images.

Saves the trained model as Trainner.yml.

TrackImages():

Recognizes students in real-time using the webcam.

Logs attendance with timestamps.

Saves attendance records in CSV files.

UpdateTable():

Updates the attendance table in the GUI with the latest records.

Usage
Run the Application:

bash
Copy
python main.py
Register Students:

Enter the student's ID and Name.

Click "Capture Images" to take facial images.

Train the Model:

Click "Train Model" after registering students.

Take Attendance:

Click "Take Attendance" to start the webcam.

The system will recognize students and log their attendance.

View Attendance:

Attendance records are displayed in the table.

Troubleshooting
Webcam Not Working:

Ensure the webcam is properly connected.

Check if other applications are using the webcam.

Missing Haar Cascade File:

Download haarcascade_frontalface_default.xml and place it in the project folder.

Low Recognition Accuracy:

Ensure proper lighting during image capture.

Capture multiple images from different angles.

CSV File Errors:

Do not manually edit CSV files.

If corrupted, delete the file and restart the application.

Future Enhancements
Export Attendance:

Add functionality to export attendance records to Excel or PDF.

Multi-User Support:

Allow multiple administrators with separate accounts.

Cloud Integration:

Store data in the cloud for remote access.

Mobile App:

Develop a mobile app for easier access.

License
This project is open-source and available under the MIT License.

Contact
For support or inquiries, please contact:

Email: support@attendance.com

GitHub: [Your GitHub Profile]

This documentation provides a comprehensive guide to using and understanding the AI-Powered Attendance System. For further details, refer to the code comments in main.py.