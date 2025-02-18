

import sys
from PyQt5.QtWidgets import QApplication
from x_gui import MainWindow

# Main function to start the application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
