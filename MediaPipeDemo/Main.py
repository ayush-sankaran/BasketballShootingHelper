import shutil
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import sys
from Threading import *
from Threading import MainWindow as MainWindow1Widget
import os

from PyQt5.QtWidgets import QWidget

class TabWidget(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project")
        self.showMaximized() 
        #self.setWindowIcon()

        tabWidget = QTabWidget()
        self.analyzerTab = MainWindow1Widget()
        self.historyTab = SecondTab()
        self.addTab = FirstTab()
        tabWidget.addTab(self.analyzerTab,"Shot Analyzer")
        tabWidget.addTab(self.historyTab,"Saved History")
        tabWidget.addTab(self.addTab,"Add Baselines")

        vbox = QVBoxLayout()
        vbox.addWidget(tabWidget)

        self.setLayout(vbox)

        
        
        tabWidget.tabBarClicked.connect(self.handle_tabbar_clicked)

    def handle_tabbar_clicked(self, index):
        if index == 0:
             self.analyzerTab.updateFileList()
        if index == 1:
            self.historyTab.loaddata()



class FirstTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MP4 File Selector')

        self.VBL = QVBoxLayout()

        self.VBL.addWidget(QLabel("Welcome to Basketball Shot Analyzer. Start by selecting a baseline Jump Shot from the drop down, or input your own baseline as an mp4.\nSelect the checkbox if you are a left handed shooter.\nOrient your body so that your entire body is visible in frame, even while jumping, and turn 45 degrees diagnoaly with your dominant hand closer to the screen.\nSelect the  Analyze Shot button. The baseline clip will play and then the webcam will open. At this point, perform a single jumpshot and hold the follow through until the webcam closes.\nYou will recieve direct feedback based on the baseline as well a similarity score from 0-1. Results will be  visualized on graphs and you can save the trail to look back at later.\n"))
        self.VBL.addStretch(1)
        self.setLayout(self.VBL)

        self.button = QPushButton('Select MP4 File to Add', self)
        self.button.setFixedWidth(300)
        self.button.clicked.connect(self.openFileDialog)
        self.VBL.addWidget(self.button)

        '''
        Welcome to Basketball Shot Analyzer. Start by selecting a baseline Jump Shot from the drop down, or input your own baseline as an mp4.\n
        Select the checkbox if you are a left handed shooter.\n
        Orient your body so that your entire body is visible in frame, even while jumping, and turn 45 degrees diagnoaly with your dominant hand closer to the screen.\n
        Select the  "Analyze Shot" button. The baseline clip will play and then the webcam will open. At this point, perform a single jumpshot and hold the follow through until the webcam closes.\n
        You will recieve direct feedback based on the baseline as well a similarity score from 0-1. Results will be  visualized on graphs and you can save the trail to look back at later.\n


        
        
        '''


    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, "Select MP4 File", "", "MP4 Files (*.mp4);;All Files (*)", options=options)
        if filePath:
            self.copyFileToCWD(filePath)

    def copyFileToCWD(self, filePath):
        cwd = os.getcwd()
        fileName = os.path.basename(filePath)
        destPath = os.path.join(cwd, fileName)
        shutil.copy(filePath, destPath)

class SecondTab(QWidget):
    def __init__(self):
        super().__init__()
        self.VBL = QVBoxLayout()
        self.label = QLabel("View Shot History", self)
        self.VBL.addWidget(self.label)
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(13)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 150)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 150)
        self.tableWidget.setColumnWidth(4, 150)
        self.tableWidget.setColumnWidth(5, 150)
        self.tableWidget.setColumnWidth(6, 250)
        self.tableWidget.setColumnWidth(7, 150)
        self.tableWidget.setColumnWidth(8, 150)
        self.tableWidget.setColumnWidth(9, 250)
        self.tableWidget.setColumnWidth(10, 150)
        self.tableWidget.setColumnWidth(11, 150)
        self.tableWidget.setColumnWidth(12, 250)
        self.tableWidget.setHorizontalHeaderLabels(["Date","Name","Comparison Player","Simalarity Score", "Min Elbow Angle", "Max Elbow Angle", "Rate of Change of Elbow Angle per Second", "Min Shoulder Angle", "Max Shoulder Angle", "Rate of Change of Shoulder Angle per Second", "Min Knee Angle", "Max Knee Angle", "Rate of Change of Knee Angle per Second"])
        self.tableWidget.setMinimumWidth(1000)
        self.tableWidget.setMinimumHeight(800)
        self.loaddata()
        self.VBL.addWidget(self.tableWidget)
        
    
    def loaddata(self):
        connection = sqlite3.connect('BasketballDatabase.db')
        cur = connection.cursor()
        sql = 'SELECT * FROM UserData'

        tablerow = 0
        results = cur.execute(sql).fetchall()
        self.tableWidget.setRowCount(len(results))

        for row in results:
            for i in range (0,13):
                if row[i] is None:
                    item = QTableWidgetItem("")
                elif isinstance(row[i], float):
                    item = QTableWidgetItem(f"{row[i]:.2f}")
                else:
                    item = QTableWidgetItem(str(row[i]))
                self.tableWidget.setItem(tablerow, i, item)
                
            tablerow+=1
        connection.close()



    

app = QApplication(sys.argv)
tabWidget = TabWidget()
tabWidget.show()
app.exec()
    
