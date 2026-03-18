import sys
from PyQt6.QtWidgets import QApplication

from view import View
from presenter import Presenter

class App(QApplication):
    def __init__(self, argv) -> None:
        super(App, self).__init__(argv)
        self.view = View()
        self.presenter = Presenter(self.view)

if __name__ == "__main__":
    app = App(sys.argv)
    app.presenter.show_ui()
    sys.exit(app.exec())