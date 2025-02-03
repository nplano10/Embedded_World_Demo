import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextBrowser, QVBoxLayout, QWidget, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt Log Example")
        self.setGeometry(100, 100, 600, 400)

        # Create the log viewer (QTextBrowser)
        self.log_viewer = QTextBrowser(self)
        
        # Create a button to trigger log messages
        self.log_button = QPushButton("Generate Log", self)
        self.log_button.clicked.connect(self.generate_log)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.log_viewer)
        layout.addWidget(self.log_button)

        # Set up central widget and layout
        container = QWidget(self)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Setup logging to display in QTextBrowser
        self.setup_logging()

    def setup_logging(self):
        # Create a custom logging handler that writes log messages to the QTextBrowser
        self.log_handler = QTextBrowserLogHandler(self.log_viewer)
        
        # Set up logging configuration
        logging.basicConfig(level=logging.DEBUG, handlers=[self.log_handler])
        logging.info("Logging system initialized.")

    def generate_log(self):
        logging.debug("This is a debug log.")
        logging.info("This is an info log.")
        logging.warning("This is a warning log.")
        logging.error("This is an error log.")
        logging.critical("This is a critical log.")

class QTextBrowserLogHandler(logging.Handler):
    def __init__(self, text_browser):
        super().__init__()
        self.text_browser = text_browser

    def emit(self, record):
        log_entry = self.format(record)
        self.text_browser.append(log_entry)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
