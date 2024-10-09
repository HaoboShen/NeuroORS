from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
import serial
import serial.tools.list_ports
import threading
import queue
import time
import csv

class MainWindow(QtWidgets.QWidget):
    # 接收数据信号
    receive_data_signal = pyqtSignal(str)
    draw_plot_signal = pyqtSignal()
    temperature_humidity_signal = pyqtSignal(list)
    # write_to_file_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle("plotter")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        # 创建四个子画图窗口
        self.plotWindow = []
        for i in range(4):
            self.plotWindow.append(pg.PlotWidget())
            self.legend = self.plotWindow[-1].addLegend()

        # 画图
        self.curves = []
        self.colors = ['red', 'orange', 'yellow', 'green', 'lightgreen','blue', 'purple', 'white']
        for i in range(32):
            pen = pg.mkPen(color=self.colors[i%8], width=1)
            curve = self.plotWindow[i//8].plot(pen=pen,name="CH%d"%(i+1))
            self.curves.append(curve)


        # 将子窗口添加到主窗口的布局中
        for i in range(4):
            self.layout.addWidget(self.plotWindow[i], i//2, i%2)

        # 初始化串口设置
        self.init_serial_settings()

        # 初始化串口工具栏
        self.init_toolbar()

        # 初始化数据接收显示
        self.init_data_display()

        # 线程结束标志
        self.receive_thread_running = False
        self.draw_thread_running = False

        # 信号连接
        self.receive_data_signal.connect(self.update_data_display)
        self.receive_data_signal.connect(self.decode_data)
        self.draw_plot_signal.connect(self.draw_plot)
        self.temperature_humidity_signal.connect(self.temperature_humidity_update)

        # 创建串口对象
        self.serial = None

        # 接收数据
        self.data_raw = ""
        self.data_str = []

        # 绘图数据
        self.data_draw_queue = queue.Queue()
        self.data_draw = []
        for i in range(32):
            self.data_draw.append([])

        # 温湿度数据
        self.data_temperature_humidity = []

        # 文件写入
        self.data_write_queue = queue.Queue()
        self.file_path = "%s.csv"%(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        # self.write_to_file_signal.connect(self.write_to_file)
        self.write_to_file_flag = False

    def init_toolbar(self):
        # 创建工具栏
        self.toolbar = QtWidgets.QToolBar(self)
        self.layout.addWidget(self.toolbar)
        self.toolbar.setStyleSheet("QToolBar{spacing:8px;}")

        # 添加串口操作按钮
        self.open_port_button = QtWidgets.QPushButton("打开串口")
        self.toolbar.addWidget(self.open_port_button)

        # 添加清楚数据按钮
        self.clear_data_button = QtWidgets.QPushButton("清除数据")
        self.toolbar.addWidget(self.clear_data_button)

        # 创建温度和湿度标签
        self.temperature_label = QtWidgets.QLabel("温度: -- °C")
        self.humidity_label = QtWidgets.QLabel("湿度: -- %")

        # 使用QHBoxLayout将标签添加到工具栏中
        self.toolbar.addWidget(self.temperature_label)
        self.toolbar.addWidget(self.humidity_label)

        # 按钮操作信号连接
        self.open_port_button.clicked.connect(self.open_serial_port)
        self.clear_data_button.clicked.connect(self.data_draw_clear)

    def init_serial_settings(self):
        # 创建串口设置布局
        serial_settings_layout = QtWidgets.QFormLayout()

        # 串口选择下拉框
        self.port_name_combobox = QtWidgets.QComboBox()
        self.port_name_combobox.addItems(self.get_available_ports())
        serial_settings_layout.addRow("串口名称:", self.port_name_combobox)

        # 波特率选择下拉框
        self.baud_rate_combobox = QtWidgets.QComboBox()
        self.baud_rate_combobox.addItems([ "115200","9600"])
        serial_settings_layout.addRow("波特率:", self.baud_rate_combobox)

        # 添加到主布局
        self.serial_settings_groupbox = QtWidgets.QGroupBox("串口设置")
        self.serial_settings_groupbox.setLayout(serial_settings_layout)
        self.layout.addWidget(self.serial_settings_groupbox)

    def init_data_display(self):
        # 创建数据显示区域
        self.data_display = QtWidgets.QPlainTextEdit()
        self.data_display.setReadOnly(True)
        self.layout.addWidget(self.data_display,3,0,1,2)

    def get_available_ports(self):
        # 获取可用串口列表
        port_list = list(serial.tools.list_ports.comports())
        if len(port_list)==0:
            return ["无可用串口"]
        else:
            port_list = [port.device for port in port_list]

        return port_list

    def open_serial_port(self):
        # 打开串口
        if self.open_port_button.text() == "关闭串口":
            self.port_name_combobox.setEnabled(True)
            self.baud_rate_combobox.setEnabled(True)
            self.close_serial_port()
            self.open_port_button.setText("打开串口")
            # 关闭线程
            self.receive_thread_running = False
            self.draw_thread_running = False
            self.write_to_file_flag = False
        else:
            self.open_port_button.setText("关闭串口")
            self.port_name_combobox.setEnabled(False)
            self.baud_rate_combobox.setEnabled(False)
            port_name = self.port_name_combobox.currentText()
            baud_rate = int(self.baud_rate_combobox.currentText())
            # 串口接收线程
            self.receive_thread = threading.Thread(target=self.receive_data)

            # 绘图线程
            self.draw_thread = threading.Thread(target=self.draw_plot_thread)

            # 文件写入线程
            self.write_to_file_thread = threading.Thread(target=self.write_to_file)
            # 创建一个串口对象
            try:
                self.serial = serial.Serial(port_name, baud_rate)
                self.update_data_display("串口已打开")
            except Exception as e:
                self.update_data_display("打开串口失败: " + str(e))
            # 创建一个线程用于接收串口数据
            try:
                self.receive_thread_running = True
                self.receive_thread.start()
            except Exception as e:
                self.update_data_display("打开接收线程失败: " + str(e))
            # 创建一个线程用于绘图
            try:
                self.draw_thread_running = True
                self.draw_thread.start()
            except Exception as e:
                self.update_data_display("打开绘图线程失败: " + str(e))
            # 创建一个线程用于写入文件
            try:
                self.write_to_file_flag = True
                self.write_to_file_thread.start()
            except Exception as e:
                self.update_data_display("打开写入文件线程失败: " + str(e))



    def close_serial_port(self):
        # 关闭串口
        if self.serial.is_open:
            # self.serial.close()
            self.receive_thread_running = False
            self.update_data_display("串口已关闭")
        else:
            self.update_data_display("串口未打开或已关闭")

    def update_data_display(self, data):
        # 更新数据显示区域
        self.data_display.appendPlainText(data)

    def receive_data(self):
        # 接收数据
        while self.receive_thread_running:
            try:
                if self.serial.in_waiting > 0 and self.serial.is_open:
                    data = self.serial.read(self.serial.in_waiting)
                    #print(data.decode(),len(data.decode()))
                    self.receive_data_signal.emit(data.decode())
            except serial.SerialException as e:
                self.update_data_display(f"接收数据时发生错误: {e}")
        # self.close_serial_port()
        self.serial.close()
        print("接收线程结束")

    def draw_plot_thread(self):
        while self.draw_thread_running:
            if self.data_draw_queue.qsize() > 0:
                # print("data_draw_queue size:",self.data_draw_queue.qsize())
                data = self.data_draw_queue.get()
                for i in range(32):
                    self.data_draw[i].append(data[i])
                    if len(self.data_draw[i]) > 1000:
                        self.data_draw[i] = self.data_draw[i][-1000:]
                self.draw_plot_signal.emit()

    def write_to_file(self):
        while self.write_to_file_flag:
            if self.data_write_queue.qsize() > 0:
                data = self.data_write_queue.get()
                # # self.write_to_file_signal.emit(data)
                # with open(self.file_path,"a") as f:
                #     f.writeline(data)
                with open(self.file_path,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

    def draw_plot(self):
        for i in range(32):
            # self.plotWindow[i//8].hideLegend()
            self.curves[i].setData(self.data_draw[i])
            # self.curves[i].setName("CH%d %f"%(i+1,self.data_draw[i][-1]))
            # self.plotWindow[i//8].showLegend() 
            # self.legend[i].setPlainText("CH%d %f"%(i+1,self.data_draw[i][-1]))
    
    def decode_data(self,data):
        # 解析数据
        self.data_raw += data
        if len(self.data_raw) > 200:
            self.data_raw = self.data_raw.replace("\r\n","")
            self.data_raw = self.data_raw.replace("\n","")
            # print("data_raw:",len(self.data_raw),"data_raw:",self.data_raw)
            self.data_str += self.data_raw.split(" ")
            self.data_str = [str for str in self.data_str if str != ""]
            # print("data_str:",self.data_str,"data_str_len:",len(self.data_str))
            self.data_raw = ""
        if len(self.data_str) >= 40 :
            if self.data_str[0] == "1" and self.data_str[1] == "1":
                data_float = list(map(float,self.data_str[2:36]))
                # if self.write_to_file_flag:
                #     self.write_to_file_signal.emit(self.data_str[:40])
                self.data_write_queue.put(self.data_str[:40])
                self.data_str = self.data_str[41:]
                self.data_temperature_humidity.clear()
                self.data_temperature_humidity.append(data_float[0])
                self.data_temperature_humidity.append(data_float[1])
                self.temperature_humidity_signal.emit(self.data_temperature_humidity)
                data_draw = data_float[2:]
                # print(data_draw)
                
                self.data_draw_queue.put(data_draw)
            else:
                self.data_str.pop(0)
    
    def temperature_humidity_update(self,data):
        self.temperature_label.setText(f"温度: {data[0]} °C")
        self.humidity_label.setText(f"湿度: {data[1]} %")
        # print("温度：",data[0],"湿度：",data[1])

    def data_draw_clear(self):
        for i in range(32):
            self.data_draw[i].clear()

    def closeEvent(self, event):
        # 关闭窗口时关闭串口
        self.receive_thread_running = False
        self.draw_thread_running = False
        self.write_to_file_flag = False
        self.close_serial_port()
        event.accept()


