import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import mediapipe as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from datetime import datetime
import os


sns.set_theme(font_scale=.5)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

# font


def angle(wrist, elbow, shoulder):
        x = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        y = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        unit_x = x / np.linalg.norm(x)
        unit_y = y / np.linalg.norm(y)
        angle_rad = np.arccos(np.dot(unit_x, unit_y))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

def getDBValues(times, angles):
        min_value = min(angles)
        max_value = max(angles)
        slope = (max_value - min_value)/( times[angles.index(max_value)] - times[angles.index(min_value)])
        return [min_value, max_value, slope]


def dtw_distance(sequence1, sequence2):
    n, m = len(sequence1), len(sequence2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(sequence1[i - 1] - sequence2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    
                                          dtw_matrix[i, j - 1],    
                                          dtw_matrix[i - 1, j - 1]) 
    
    return dtw_matrix[n, m]
    

def normalize_sequence(sequence):
    sequence = np.array(sequence)
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    return (sequence - min_val) / (max_val - min_val)

def similarity_measure(sequence1, sequence2):

    seq1 = normalize_sequence(np.array(sequence1))
    seq2 = normalize_sequence(np.array(sequence2))
    
    distance = dtw_distance(seq1, seq2)
    
    max_length = max(len(seq1), len(seq2))
    normalized_distance = distance / max_length
    
    similarity = 1 / (1 + normalized_distance)
    
    return similarity

def ovr_similarity(df1, df2):

    x = similarity_measure(df1["Elbow Angle"], df2["Elbow Angle"])
    y = similarity_measure(df1["Shoulder Angle"], df2["Shoulder Angle"])
    z = similarity_measure(df1["Knee Angle"], df2["Knee Angle"])

    return (x+y+z)/3

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.showMaximized() 

        #Vertical Layout
        self.VBL = QVBoxLayout()

        # Filename DropDown
        self.VBL.addWidget(QLabel("Select A Baseline"))
        self.FileNames = QComboBox()
        self.updateFileList()
        self.FileNames.setFixedWidth(200)
        self.VBL.addWidget(self.FileNames)

        # Lefty Label + CheckBox
        self.HBL = QHBoxLayout()
        self.leftyBTN = QCheckBox()
        self.HBL.addWidget(QLabel("Lefty"))
        self.HBL.addWidget(self.leftyBTN)
        self.HBL.addStretch()  
        self.VBL.addLayout(self.HBL)
        self.VBL.addWidget(self.leftyBTN)

        # Video Area
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        # Run Analysis Button
        self.StartBTN = QPushButton("Analyze Shot")
        self.StartBTN.setFixedWidth(350)
        self.StartBTN.clicked.connect(self.StartFeed)
        self.VBL.addWidget(self.StartBTN)

        # Set up Canvas for Graphs
        self.Canvas =  PlotCanvas(self)
        self.VBL.addWidget(self.Canvas)
        #self.Canvas.hide()


        # Edit text for save Name
        self.Title = QLineEdit()
        self.Title.setPlaceholderText("Save attempt as... ")
        self.Title.setFixedWidth(300)
        self.VBL.addWidget(self.Title)

        #Save Button
        self.SaveBTN = QPushButton("Save")
        self.SaveBTN.clicked.connect(self.saveToDB)
        self.SaveBTN.setFixedWidth(100)
        self.SaveBTN.setEnabled(False)  # Disable the Save button until the process is done
        self.VBL.addWidget(self.SaveBTN)


        self.Worker1 = Worker1(self.Canvas, self.leftyBTN)

        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.Worker1.processFinished.connect(self.setUserDBValues)

        self.Worker1.feedbackSignal.connect(self.showToast)

        self.Worker1.finished.connect(self.clearFeedLabel)

        self.VBL.addStretch(1)
        self.setLayout(self.VBL)

        self.user_db_values = None  
        self.similarityScore = None
        self.compiPlayer = None


    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def updateFileList(self):
        self.FileNames.clear()
        mp4_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
        self.FileNames.addItems(mp4_files)

    def showToast(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Shot Feedback")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.buttonClicked.connect(msg.close)
        msg.exec_()

    
    def clearFeedLabel(self):
        self.FeedLabel.clear()

    def StartFeed(self):
        #self.Canvas.hide()
        self.Worker1.setFileName(self.FileNames.currentText())
        self.Worker1.start()

    def saveToDB(self):
        name = self.Title.text()
        if self.user_db_values is not None and self.similarityScore is not None and self.compiPlayer is not None:
            addToDB(name, self.user_db_values, self.similarityScore, self.compiPlayer)
        self.showSuccessMessage()
    
    def setUserDBValues(self, user_db_values, score, c):
        self.user_db_values = user_db_values
        self.similarityScore = score
        self.compiPlayer = c
        self.SaveBTN.setEnabled(True)  # Enable the Save button when the process is done

    def showSuccessMessage(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Successfully added to the database.")
        msg.setWindowTitle("Success")
        msg.exec_()


class Worker1(QThread):

    ImageUpdate = pyqtSignal(QImage)
    processFinished = pyqtSignal(object,float, str)
    feedbackSignal = pyqtSignal(str)

    def __init__(self, canvas,leftyBT, parent = None):
        super().__init__(parent)
        self.graph = canvas
        self.leftyBT = leftyBT
        self.fileName = None

    def setFileName(self, fileName):
        self.fileName = fileName

    def run(self):

        baseline_df, alpha, baseline_db_values = self.get_data(filename = self.fileName)
        user_df, uselss, user_db_values = self.get_data(a = alpha, lefty = self.leftyBT.isChecked())

        
        self.graph.clear_plot()
        self.graph.plot(user_df, baseline_df)
        #self.graph.show()
        self.score = ovr_similarity(baseline_df,user_df)
        #print(self.score)
        self.processFinished.emit(user_db_values,self.score, self.fileName)


        user_feedback = []

        conn = sqlite3.connect("BasketballDatabase.db")
        cur = conn.cursor()
        #Elbow min
        if user_db_values["Elbow"][0] - baseline_db_values["Elbow"][0] > 12:
            user_feedback.append(cur.execute("SELECT elbowMin FROM Conclusions WHERE accuracy = 'Positive'").fetchone())
        if user_db_values["Elbow"][0] - baseline_db_values["Elbow"][0] < -12:
            user_feedback.append(cur.execute("SELECT elbowMin FROM Conclusions WHERE accuracy = 'Negative'").fetchone())
        #Elbow Max
        if user_db_values["Elbow"][1] - baseline_db_values["Elbow"][1] > 9:
            user_feedback.append(cur.execute("SELECT elbowMax FROM Conclusions WHERE accuracy = 'Positive' ").fetchone())
        if user_db_values["Elbow"][1] - baseline_db_values["Elbow"][1] < -9:
            user_feedback.append(cur.execute("SELECT elbowMax FROM Conclusions WHERE accuracy = 'Negative' ").fetchone())
        #Elbow slope
        if user_db_values["Elbow"][2] - baseline_db_values["Elbow"][2] > .3:
            user_feedback.append(cur.execute("SELECT elbowSlope FROM Conclusions WHERE accuracy = 'Positive' ").fetchone())
        if user_db_values["Elbow"][2] - baseline_db_values["Elbow"][2] < -.3:
            user_feedback.append(cur.execute("SELECT elbowSlope FROM Conclusions WHERE accuracy = 'Negative' ").fetchone())

        #Shoulder min
        if user_db_values["Shoulder"][0] - baseline_db_values["Shoulder"][0] > 12:
            user_feedback.append(cur.execute("SELECT shoulderMin FROM Conclusions WHERE accuracy = 'Positive' ").fetchone())
        if user_db_values["Shoulder"][0] - baseline_db_values["Shoulder"][0] < -12:
            user_feedback.append(cur.execute("SELECT shoulderMin FROM Conclusions WHERE accuracy = 'Negative' ").fetchone())
        #Shoudler Max
        if user_db_values["Shoulder"][1] - baseline_db_values["Shoulder"][1] > 9:
            user_feedback.append(cur.execute("SELECT shoulderMax FROM Conclusions WHERE accuracy = 'Positive' ").fetchone())
        if user_db_values["Shoulder"][1] - baseline_db_values["Shoulder"][1] < -9:
            user_feedback.append(cur.execute("SELECT shoulderMax FROM Conclusions WHERE accuracy = 'Negative'").fetchone())
        #Shoulder slope
        if user_db_values["Shoulder"][2] - baseline_db_values["Shoulder"][2] > .3:
            user_feedback.append(cur.execute("SELECT shoulderSlope FROM Conclusions WHERE accuracy ='Positive'").fetchone())
        if user_db_values["Shoulder"][2] - baseline_db_values["Shoulder"][2] < -.3:
            user_feedback.append(cur.execute("SELECT shoulderSlope FROM Conclusions WHERE accuracy = 'Negative'").fetchone())

        #Knee min
        if user_db_values["Knee"][0] - baseline_db_values["Knee"][0] > 12:
            user_feedback.append(cur.execute("SELECT kneeMin FROM Conclusions WHERE accuracy = 'Positive'").fetchone())
        if user_db_values["Knee"][0] - baseline_db_values["Knee"][0] < -12:
            user_feedback.append(cur.execute("SELECT kneeMin FROM Conclusions WHERE accuracy = 'Negative'").fetchone())
        #Knee Max
        if user_db_values["Knee"][1] - baseline_db_values["Knee"][1] > 9:
            user_feedback.append(cur.execute("SELECT kneeMax FROM Conclusions WHERE accuracy = 'Positive'").fetchone())
        if user_db_values["Knee"][1] - baseline_db_values["Knee"][1] < -9:
            user_feedback.append(cur.execute("SELECT kneeMax FROM Conclusions WHERE accuracy = 'Negative'").fetchone())
        #Knee slope
        if user_db_values["Knee"][2] - baseline_db_values["Knee"][2] > .3:
            user_feedback.append(cur.execute("SELECT kneeSlope FROM Conclusions WHERE accuracy = 'Positive'").fetchone())
        if user_db_values["Knee"][2] - baseline_db_values["Knee"][2] < -.3:
            user_feedback.append(cur.execute("SELECT kneeSlope FROM Conclusions WHERE accuracy = 'Negative'").fetchone())

        #print(user_feedback)

        feedback_messages = [f[0] for f in user_feedback if f is not None]
        feedback_text = "\n\n".join(feedback_messages)
        feedback_text = feedback_text + "\n\n Similarity score: " + f"{self.score:.2f}"
        self.feedbackSignal.emit(feedback_text)



    def stop(self):
        self.ThreadActive = False
        self.quit()
    
    
    def get_data(self, filename = None, lefty=False, a = float('inf')):
        if filename is None:
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(filename)
        
        previous = time()
        delta = 0
        duration = 0
        time_values = []
        elbow_angles = []
        knee_angles = []
        shoulder_angles = []

        with mp_holistic.Holistic(static_image_mode=True) as holistic:
            while True and duration < a:
                ret, frame = cap.read()
                if ret:
                    result = holistic.process(frame)
                    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    current = time()
                    delta += current - previous
                    previous = current

                    # Check if 3 (or some other value) seconds passed
                    if delta > .1:
                        duration += delta
                        if result.pose_landmarks:
                            if lefty:
                                wrist = result.pose_landmarks.landmark[15]
                                elbow = result.pose_landmarks.landmark[13]
                                shoulder = result.pose_landmarks.landmark[11]
                                ankle = result.pose_landmarks.landmark[27]
                                knee = result.pose_landmarks.landmark[25]
                                hip = result.pose_landmarks.landmark[23]
                            
                            
                            else:
                                wrist = result.pose_landmarks.landmark[16]
                                elbow = result.pose_landmarks.landmark[14]
                                shoulder = result.pose_landmarks.landmark[12]
                                ankle = result.pose_landmarks.landmark[28]
                                knee = result.pose_landmarks.landmark[26]
                                hip = result.pose_landmarks.landmark[24]

                            time_values.append(duration)
                            elbow_angles.append(angle(wrist, elbow, shoulder))
                            knee_angles.append(angle(ankle, knee, hip))
                            shoulder_angles.append(angle(elbow,shoulder,hip))

                            delta = 0

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if filename is None:
                        frame = cv2.flip(frame, 1)
                    ConvertToQtFormat = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.ImageUpdate.emit(Pic)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        break
                else:
                    cap.release()
                    break
        
        
        if filename is None:
            data_points = {'Time': time_values, 'Elbow Angle': elbow_angles, 'Knee Angle' : knee_angles, 'Shoulder Angle': shoulder_angles, 'ID': "user"}
        else:
            data_points = {'Time': time_values, 'Elbow Angle': elbow_angles, 'Knee Angle' : knee_angles, 'Shoulder Angle': shoulder_angles, 'ID' :'baseline'}
        database_values = {'Elbow': getDBValues(time_values, elbow_angles) ,  'Shoulder': getDBValues(time_values, shoulder_angles), 'Knee' : getDBValues(time_values, knee_angles)}
        df = pd.DataFrame(data=data_points)
        df2 = pd.DataFrame(data = database_values)
        return df,duration,df2
    
class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=6, height=4):

        fig, self.axes = plt.subplots(1, 3, figsize=(6, 4))
        self.axes = self.axes.flatten()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        #self.plot()

    def clear_plot(self):
        for ax in self.axes:
            ax.clear()  # Clear each axis
        self.draw()  # Redraw the canvas

    def plot(self, user_df, baseline_df):

        sns.lineplot(x='Time', y='Elbow Angle', data=user_df, ax=self.axes[0], label='User')
        sns.lineplot(x='Time', y='Elbow Angle', data=baseline_df, ax=self.axes[0], label='Baseline')
        self.axes[0].set_title('Elbow Angles')

        sns.lineplot(x='Time', y='Shoulder Angle', data=user_df, ax=self.axes[1], label='User')
        sns.lineplot(x='Time', y='Shoulder Angle', data=baseline_df, ax=self.axes[1], label='Baseline')
        self.axes[1].set_title('Shoulder Angles')

        sns.lineplot(x='Time', y='Knee Angle', data=user_df, ax=self.axes[2], label='User')
        sns.lineplot(x='Time', y='Knee Angle', data=baseline_df, ax=self.axes[2], label='Baseline')
        self.axes[2].set_title('Knee Angles')
        self.draw()


def addToDB(name, df, score, compPlayer = "CurryForm.mp4"):
    conn = sqlite3.connect("BasketballDatabase.db")
    cur = conn.cursor()
    cur.execute('insert into UserData (Date, Name, comparisonPlayer, similarityScore, elbowMin, elbowMax, elbowSlope, shoulderMin, shoulderMax, shoulderSlope, kneeMin, kneeMax, kneeSlope) values (?,?,?,?,?,?,?,?,?,?,?,?,?)', (datetime.now().strftime("%m/%d/%Y %H:%M"), name, compPlayer, score, df["Elbow"][0],df["Elbow"][1],df["Elbow"][2],df["Shoulder"][0],df["Shoulder"][1],df["Shoulder"][2],df["Knee"][0],df["Knee"][1],df["Knee"][2]))
    #cur.execute('insert into UserData (Date, Name, similarityScore, elbowMin, elbowMax, elbowSlope, shoulderMin, shoulderMax, shoulderSlope, kneeMin, kneeMax, kneeSlope) values (?,?,?,?,?,?,?,?,?,?,?,?)', (datetime.now().strftime("%m/%d/%Y %H:%M"), name, score, df["Elbow"][0],df["Elbow"][1],df["Elbow"][2],df["Shoulder"][0],df["Shoulder"][1],df["Shoulder"][2],df["Knee"][0],df["Knee"][1],df["Knee"][2]))
                                                                                                                                                                                         #datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())