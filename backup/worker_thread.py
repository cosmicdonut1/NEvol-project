from PyQt5.QtCore import QThread, pyqtSignal


class WorkerThread(QThread):
    finished = pyqtSignal()
    resultReady = pyqtSignal(Object)  # Define any data type you need

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        # Heavy computations or data processing
        result = self.do_work()

        # Emit signal with result
        self.resultReady.emit(result)
        self.finished.emit()

    def do_work(self):
        # Implement your heavy computations here
        import time
        time.sleep(5)  # Simulate heavy work
        return "Some result"