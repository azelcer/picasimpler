import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from picasimpler.main.view import View
from picasimpler.main.presenter import Presenter

class App(QApplication):
    def __init__(self, argv) -> None:
        super(App, self).__init__(argv)
        # MVP Pattern
        self.view = View()
        self.presenter = Presenter(self.view)

if __name__ == "__main__":
    app = App(sys.argv)
    # setting app style
    style_file: Path = Path("./src/picasimpler/resources/main.css")
    with open(style_file, "r") as st_f:
        st_f_content = st_f.read()
    app.setStyleSheet(st_f_content)
    app.presenter.show_ui()
    sys.exit(app.exec())