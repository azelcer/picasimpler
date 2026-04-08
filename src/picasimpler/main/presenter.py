from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import logging as _lgn
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog
from functools import wraps

from picasimpler.main.view import View
from picasimpler.main.analysis import AnalysisWorker
from picasimpler.helpers.status import AnalysisStatus, UIColor, MessageType
from picasimpler.helpers.utils import safe_float_to0, safe_float_tonone
from picasimpler.config.config_var import (
    SPAT_TOL_NM_DEF,
    SPAT_TOL_NM_MIN,
    PRECLUST_GAMMA_DEF,
    MAX_PRECLUST_GAMMA,
    PRECLUST_EPS_DEF,
    LAMDBA_EXC_DEF,
    LAMBDA_EM_DEF,
    LAMBDA_MIN,
    NA_IDX_DEF,
    NI_DEF,
    NS_DEF,
    RES_DIR,
    Z_SIM_FIT_ARR
)

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)


class PresenterSignals(QObject):
    request_load_data = pyqtSignal(object, object)
    request_start_filtering = pyqtSignal()
    request_start_clustering = pyqtSignal()
    request_calibration = pyqtSignal()
    request_calibration_fromfile = pyqtSignal(object)
    request_refit_origami = pyqtSignal(int)
    send_spat_tol_toanalysis = pyqtSignal(float)
    send_preclust_gamma_toanalysis = pyqtSignal(float)
    send_preclust_eps_toanalysis = pyqtSignal(float)
    send_n_guess_toanalysis = pyqtSignal(object)
    send_n_guess_choice_toanalysis = pyqtSignal(bool)
    send_lambda_exc_toanalysis = pyqtSignal(float)
    send_lambda_em_toanalysis = pyqtSignal(float)
    send_n_s_toanalysis = pyqtSignal(float)
    send_n_i_toanalysis = pyqtSignal(float)
    send_coll_fl_tab_toanalysis = pyqtSignal(object)

class Presenter(QObject):
    """
    this class manages connections and signals between frontend and backend
    """
    def __init__(self, view: View):
        super().__init__()
        self._view = view
        self.signals = PresenterSignals()
        # signal-slot connections with UI
        self._make_ui_connect()
        # analysis worker, thread and connection preparation
        self.start_analysis_thread()

        # initialize some variables
        self.tot_elem_curr_analysis_step: int | None = None
        self.curr_displ_orig_num: int | None = None
        self.analysis_status: AnalysisStatus = AnalysisStatus.PRE_ANALYSIS
        self.simpler_tol:float = SPAT_TOL_NM_DEF
        self.preclust_gamma: float = PRECLUST_GAMMA_DEF
        self.preclust_eps: float = PRECLUST_EPS_DEF
        self.n_guess = (None, None, None, None)
        self.should_use_n_guess = False
        self.lambda_exc: float = LAMDBA_EXC_DEF
        self.lambda_em: float = LAMBDA_EM_DEF
        self.n_s: float = NS_DEF
        self.n_i: float = NI_DEF
        self.res_dir: Path = RES_DIR
        self._view.ui.NA_combobox.setCurrentIndex(NA_IDX_DEF)
        self._view.ui.NA_combobox.activated.emit(self._view.ui.NA_combobox.currentIndex())
        
    def check_analysis_status(ref_analysis_status: AnalysisStatus):
        """
        This decorator wraps a function in a if statement that gets executed only if the
        analysis status has passed the reference step given as an argument.
        Optionally, if must_not_analysing is True, it wraps with another if statement that checks
        whether the program is not analyzing right now
        """
        def check_analysis_status_innderdecor(decorated_func):
            @wraps(decorated_func)
            def wrapper_func(self: Presenter, *args, **kwargs):
                if self.analysis_status.passed_analysis_step(ref_analysis_status) and not self.analysis_status.is_analysing:
                    return decorated_func(self, *args, **kwargs)
                else:
                    return
            return wrapper_func
        return check_analysis_status_innderdecor
        
    @property
    def analysis_status(self):
        return self._analysis_status
    
    @analysis_status.setter
    def analysis_status(self, status: AnalysisStatus):
        self._analysis_status = status
        self._view.upd_analysis_status_onui(status)
        
    @property
    def simpler_tol(self):
        return self._simpler_tol
    
    @simpler_tol.setter
    def simpler_tol(self, value: float):
        self._simpler_tol = max((value, SPAT_TOL_NM_MIN))
        self.signals.send_spat_tol_toanalysis.emit(self._simpler_tol)
        self._view.upd_spat_tol_onui(self._simpler_tol)
        
    @property
    def preclust_gamma(self):
        return self._preclust_gamma
    
    @preclust_gamma.setter
    def preclust_gamma(self, value: float):
        self._preclust_gamma = min((value, MAX_PRECLUST_GAMMA))
        self.signals.send_preclust_gamma_toanalysis.emit(self._preclust_gamma)
        self._view.upd_preclust_gamma_onui(self._preclust_gamma)
        
    @property
    def preclust_eps(self):
        return self._preclust_eps
    
    @preclust_eps.setter
    def preclust_eps(self, value: float):
        self._preclust_eps = value
        self.signals.send_preclust_eps_toanalysis.emit(self._preclust_eps)
        self._view.upd_preclust_eps_onui(self._preclust_eps)
        
    @property
    def n_guess(self):
        return self._n_guess
    
    @n_guess.setter
    def n_guess(self, value: float):
        self._n_guess = value
        self.signals.send_n_guess_toanalysis.emit(self._n_guess)
        
    @property
    def should_use_n_guess(self):
        return self._should_use_n_guess
    
    @should_use_n_guess.setter
    def should_use_n_guess(self, value: bool):
        self._should_use_n_guess = value
        self.signals.send_n_guess_choice_toanalysis.emit(self._should_use_n_guess)
        
    @property
    def lambda_em(self):
        return self._lambda_em
    
    @lambda_em.setter
    def lambda_em(self, value: float):
        self._lambda_em = max((value, LAMBDA_MIN))
        self.signals.send_lambda_em_toanalysis.emit(self._lambda_em)
        self._view.upd_lambda_em_onui(self._lambda_em)
        
    @property
    def lambda_exc(self):
        return self._lambda_exc
    
    @lambda_exc.setter
    def lambda_exc(self, value: float):
        self._lambda_exc = max((value, LAMBDA_MIN))
        self.signals.send_lambda_exc_toanalysis.emit(self._lambda_exc)
        self._view.upd_lambda_exc_onui(self._lambda_exc)
        
    @property
    def n_s(self):
        return self._n_s
    
    @n_s.setter
    def n_s(self, value: float):
        self._n_s = max((value, 1))
        self.signals.send_n_s_toanalysis.emit(self._n_s)
        self._view.upd_n_s_onui(self._n_s)
        
    @property
    def n_i(self):
        return self._n_i
    
    @n_i.setter
    def n_i(self, value: float):
        self._n_i = max((value, self.n_s + 0.01))
        self.signals.send_n_i_toanalysis.emit(self._n_i)
        self._view.upd_n_i_onui(self._n_i)
        
    def show_ui(self):
        self._view.show()
        
    def _make_ui_connect(self):
        """
        this functions makes all the connection with the signals coming from the UI
        """
        # connect signals from UI to presenter
        # analysis buttons
        self._view.ui.browse_file_button.clicked.connect(self._browse_file)
        self._view.ui.filter_button.clicked.connect(self._start_filtering)
        self._view.ui.cluster_button.clicked.connect(self._start_clustering)
        self._view.ui.save_clust_button.clicked.connect(self._order_save_clust)
        self._view.ui.calib_button.clicked.connect(self._order_calibration)
        self._view.ui.calib_fromfile_button.clicked.connect(self._order_calibration_fromfile)
        # origami navigation buttons
        self._view.ui.next_orig_button.clicked.connect(self._order_plot_next_orig)
        self._view.ui.prev_orig_button.clicked.connect(self._order_plot_prev_orig)
        self._view.ui.disc_selec_orig_button.clicked.connect(self._disc_selec_orig)
        self._view.ui.refit_origami.clicked.connect(self._order_refit_orig)
        # parameters inputs from UI
        self._view.ui.simpler_spat_tol_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_spat_tol_fromui.emit(
                safe_float_to0(self._view.ui.simpler_spat_tol_lineedit.text())
            )
        )
        self._view.ui.preclust_gamma_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_preclust_gamma_fromui.emit(
                safe_float_to0(self._view.ui.preclust_gamma_lineedit.text())
            )
        )
        self._view.ui.preclust_eps_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_preclust_eps_fromui.emit(
                safe_float_to0(self._view.ui.preclust_eps_lineedit.text())
            )
        )
        self._view.ui.n1_guess_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_n_guess_fromui.emit((
                safe_float_tonone(self._view.ui.n1_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n2_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n3_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n4_guess_lineedit.text())    
            ))
        )
        self._view.ui.n2_guess_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_n_guess_fromui.emit((
                safe_float_tonone(self._view.ui.n1_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n2_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n3_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n4_guess_lineedit.text())    
            ))
        )
        self._view.ui.n3_guess_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_n_guess_fromui.emit((
                safe_float_tonone(self._view.ui.n1_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n2_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n3_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n4_guess_lineedit.text())    
            ))
        )
        self._view.ui.n4_guess_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_n_guess_fromui.emit((
                safe_float_tonone(self._view.ui.n1_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n2_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n3_guess_lineedit.text()),
                safe_float_tonone(self._view.ui.n4_guess_lineedit.text())    
            ))
        )
        self._view.ui.manual_guess_checkbox.stateChanged.connect(
            lambda: self._view.signals.send_n_guess_choice_fromui.emit(
                self._view.ui.manual_guess_checkbox.isChecked()
            )
        )
        self._view.ui.lambda_exc_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_lambda_exc_fromui.emit(
                safe_float_to0(self._view.ui.lambda_exc_lineedit.text())
            )
        )
        self._view.ui.lambda_em_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_lambda_em_fromui.emit(
                safe_float_to0(self._view.ui.lambda_em_lineedit.text())
            )
        )
        self._view.ui.n_s_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_n_s_fromui.emit(
                safe_float_to0(self._view.ui.n_s_lineedit.text())
            )
        )
        self._view.ui.n_i_lineedit.manual_editing_finished.connect(
            lambda: self._view.signals.send_n_i_fromui.emit(
                safe_float_to0(self._view.ui.n_i_lineedit.text())
            )
        )
        self._view.signals.send_spat_tol_fromui.connect(self.upd_spat_tol)
        self._view.signals.send_preclust_gamma_fromui.connect(self.upd_preclust_gamma)
        self._view.signals.send_preclust_eps_fromui.connect(self.upd_preclust_eps)
        self._view.signals.send_n_guess_fromui.connect(self.upd_n_guess)
        self._view.signals.send_n_guess_choice_fromui.connect(self.upd_n_guess_choice)
        self._view.signals.send_lambda_exc_fromui.connect(self.upd_lambda_exc)
        self._view.signals.send_lambda_em_fromui.connect(self.upd_lambda_em)
        self._view.signals.send_n_s_fromui.connect(self.upd_n_s)
        self._view.signals.send_n_i_fromui.connect(self.upd_n_i)
        self._view.ui.NA_combobox.activated.connect(self.upd_NA)
        
    def _make_analysis_connect(self):
        """
        this functions makes all the connection with the signals coming from the analysis thread
        """
        # connect signals from presenter to analysis worker
        self.signals.request_load_data.connect(self._analysis_worker.load_data)
        self.signals.request_start_filtering.connect(self._analysis_worker.do_filt)
        self.signals.request_start_clustering.connect(self._analysis_worker.do_clust)
        self.signals.request_calibration.connect(self._analysis_worker.do_calib)
        self.signals.request_calibration_fromfile.connect(self._analysis_worker.do_calib_fromfile)
        self.signals.send_spat_tol_toanalysis.connect(self._analysis_worker.upd_spat_tol)
        self.signals.send_preclust_gamma_toanalysis.connect(self._analysis_worker.upd_preclust_gamma)
        self.signals.send_preclust_eps_toanalysis.connect(self._analysis_worker.upd_preclust_eps)
        self.signals.send_n_guess_toanalysis.connect(self._analysis_worker.upd_n_guess)
        self.signals.send_n_guess_choice_toanalysis.connect(self._analysis_worker.upd_n_guess_choice)
        self.signals.send_lambda_exc_toanalysis.connect(self._analysis_worker.upd_lambda_exc)
        self.signals.send_lambda_em_toanalysis.connect(self._analysis_worker.upd_lambda_em)
        self.signals.send_n_i_toanalysis.connect(self._analysis_worker.upd_n_i)
        self.signals.send_n_s_toanalysis.connect(self._analysis_worker.upd_n_s)
        self.signals.send_coll_fl_tab_toanalysis.connect(self._analysis_worker.upd_coll_fl_tab)
        
        # connect signals from analysis worker to presenter
        self._analysis_worker.signals.tell_data_loaded.connect(self._on_data_loaded)
        self._analysis_worker.signals.tell_analysis_step_start.connect(self._on_new_analysis_step)
        self._analysis_worker.signals.tell_filt_done.connect(self._on_filt_done)
        self._analysis_worker.signals.tell_clust_done.connect(self._on_clust_done)
        self._analysis_worker.signals.tell_refit_done.connect(self._on_refit_done)
        self._analysis_worker.signals.send_msg_toprint.connect(self._print_to_ui)
        self._analysis_worker.signals.tell_calib_done.connect(self._on_calib_done)
        # connect signals from other analysis helper classes to presenter
        self._analysis_worker.simpler_signals.tell_analysis_elem_done.connect(self._on_new_analysis_elem)
        self._analysis_worker.simpler_signals.send_msg_toprint.connect(self._print_to_ui)
        self._analysis_worker.clust_signals.tell_analysis_elem_done.connect(self._on_new_analysis_elem)
        self._analysis_worker.clust_signals.send_msg_toprint.connect(self._print_to_ui)
        # connect signals from presenter to other analysis helper classes
        self.signals.request_refit_origami.connect(self._analysis_worker.refit_orig)
        
    def start_analysis_thread(self):
        """
        This function creates an empty analysis worker, makes all signal-slot connections and moves it to another thread
        """
        self._analysis_worker = AnalysisWorker()
        self._analysis_thread = QThread()
        self._make_analysis_connect()
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.start()
        
    def close_analysis_thread(self):
        """
        This function closes the analysis thread
        """
        self._analysis_thread.quit()
        self._analysis_thread.wait()
        
    @pyqtSlot(MessageType, str)
    def _print_to_ui(self, msg_type: MessageType, msg_toprint: str):
        '''
        This function prints a message on the message box in the UI
        '''
        self._view.upd_msg_onui(msg_type, msg_toprint)
        
    @pyqtSlot()
    def _browse_file(self):
        """
        this function opens a window to choose the hdf5 file used for calibration
        """
        filepath_str, _ = QFileDialog.getOpenFileName(
            self._view,
            directory=str(Path.home()), # home directory, OS independent
            caption="Select calibration file",
            filter="(*.hdf5)"
        )
        if filepath_str:
            self.data_path = Path(filepath_str)
            self.metadata_path = self.data_path.parent / Path(self.data_path.stem + ".yaml")
            self._order_load_data()
            
    def _order_load_data(self):
        """
        This function starts a new analysis thread, creates a new analysis worker and connects its signals
        to the Presenter
        """
        self.analysis_status = AnalysisStatus.PRE_ANALYSIS
        self._view.reset_ui()
        self.signals.request_load_data.emit(self.data_path, self.metadata_path)
            
    @pyqtSlot()
    def _on_data_loaded(self):
        """
        This function is called upon successfull data loading.
        It prints on UI the file name and folder, and updates the analysis status.
        """
        self._view.upd_data_file_onui(self.data_path)
        self.analysis_status = AnalysisStatus.DATA_LOADED
            
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.DATA_LOADED)
    def _start_filtering(self):
        """
        this function tells the analysis worker to perform all the filtering steps needed before site clustering
        """
        self.signals.request_start_filtering.emit()
    
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.FILT_DONE)
    def _start_clustering(self):
        """
        This function tells the analysis worker to start site clusterization procedure
        """
        if self.should_use_n_guess and all(guess is not None for guess in self.n_guess):
            self._print_to_ui(MessageType.WARNING, "Doing batch clustering using photon number guesses. It is recommended to use it only for single origami re-fit.")
        self.signals.request_start_clustering.emit()
            
    @pyqtSlot(AnalysisStatus, int)
    def _on_new_analysis_step(self, analysis_status: AnalysisStatus, tot_elem_curr_analysis_step: int):
        """
        this function is called whenever a new analysis step is started by the analysis worker.
        It tells the View to update the analysis status on UI, and it updates the total number of
        elements in the current analysis step
        """
        self.analysis_status = analysis_status
        self.tot_elem_curr_analysis_step = tot_elem_curr_analysis_step
        self._view.upd_analysis_counter(0, self.tot_elem_curr_analysis_step)
        
    @pyqtSlot(int)
    def _on_new_analysis_elem(self, elem_num: int):
        """
        This function is called everytime a new individual element is analyzed within an
        analysis step. It tells the View to update the counter on the UI
        """
        self._view.upd_analysis_counter(elem_num, self.tot_elem_curr_analysis_step)
        
    @pyqtSlot()
    def _on_filt_done(self):
        """
        This function is called once all the filtering steps are done.
        It displays the first origami scatter plot without clustering
        """
        self.analysis_status = AnalysisStatus.FILT_DONE
        self.curr_displ_orig_num = 0
        self._view.plot_orig(self._analysis_worker.simpler, self.curr_displ_orig_num)
        self._view.upd_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.tot_orig)
        self._view.set_color_frame(UIColor.GRAY)

    @pyqtSlot(bool)
    def _on_clust_done(self, are_there_clust):
        """
        This function is called once the site clusterization is completed.
        It displayed the first origami scatter plot with clustering
        """
        if not are_there_clust:
            self.analysis_status = AnalysisStatus.FILT_DONE
            _lgr.warning("No valid clusters found")
            self._print_to_ui(MessageType.WARNING, "No valid clusters found")
            return
        else:
            self.analysis_status = AnalysisStatus.CLUST_DONE
            self.curr_displ_orig_num = 0
            self._view.plot_orig_wclust(
                self._analysis_worker.simpler,
                self._analysis_worker.clust,
                self.curr_displ_orig_num,
                self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num]
            )
            self._view.upd_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.clust.tot_orig_kept)
            self._view.upd_selec_orig_counter(self._analysis_worker.clust.selec_orig_list)

    def _do_plot_shift(self, shift: int):
        if self.analysis_status.passed_analysis_step(AnalysisStatus.CLUST_DONE):
            self.curr_displ_orig_num += shift
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.clust.tot_orig_kept
            self._view.upd_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.clust.tot_orig_kept)
            self._view.plot_orig_wclust(
                self._analysis_worker.simpler,
                self._analysis_worker.clust,
                self.curr_displ_orig_num,
                self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num]
            )
        elif self.analysis_status.passed_analysis_step(AnalysisStatus.FILT_DONE):
            self.curr_displ_orig_num += shift
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.tot_orig
            self._view.upd_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.tot_orig)
            self._view.plot_orig(self._analysis_worker.simpler, self.curr_displ_orig_num)

    @pyqtSlot()
    def _order_plot_next_orig(self):
        self._do_plot_shift(1)

    @pyqtSlot()
    def _order_plot_prev_orig(self):
        self._do_plot_shift(-1)
        
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.CLUST_DONE)
    def _disc_selec_orig(self):
        """
        This function discards/select an origami based on its current state,
        and update the corresponsing button and frame color
        """
        if self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num]:
            self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num] = False
            self._view.set_color_frame(UIColor.MAG)
            self._view.upd_discard_button_toselec()
            self._view.upd_selec_orig_counter(self._analysis_worker.clust.selec_orig_list)
        else:
            self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num] = True
            self._view.set_color_frame(UIColor.G)
            self._view.upd_selec_button_todiscard()
            self._view.upd_selec_orig_counter(self._analysis_worker.clust.selec_orig_list)
           
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.CLUST_DONE)
    def _order_refit_orig(self):
        self.signals.request_refit_origami.emit(self.curr_displ_orig_num)
        
    @pyqtSlot()
    def _on_refit_done(self):
        self._do_plot_shift(0)
            
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.CLUST_DONE)
    def _order_save_clust(self):
        """
        This function, if clusterization is done, orders the analysis worker to save clusterization results
        in a .npy file
        """
        self._analysis_worker.save_clust()
        
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.CLUST_DONE)
    def _order_calibration(self):
        """
        This function, if clusterization is done, orders the analysis worker to perform the final
        SIMPLER calibration
        """
        self.signals.request_calibration.emit()
        
    @pyqtSlot()
    def _order_calibration_fromfile(self):
        """
        This function opens a window to choose a .npy file containing the results of a clusterization and, if the file is opened successfully and has
        the expected structure, it orders the analysis worker to perform the final SIMPLER calibration using the uploaded data
        """
        filepath_str, _ = QFileDialog.getOpenFileName(
            self._view,
            directory=str(Path.home()), # home directory, OS independent
            caption="Select calibration file",
            filter="(*.npy)"
        )
        if filepath_str:
            self.clust_res_filepath = Path(filepath_str)
            self.signals.request_calibration_fromfile.emit(self.clust_res_filepath)
        
    @pyqtSlot(float)
    def upd_spat_tol(self, value):
        self.simpler_tol = value
        
    @pyqtSlot(float)
    def upd_preclust_gamma(self, value):
        self.preclust_gamma = value
        
    @pyqtSlot(float)
    def upd_preclust_eps(self, value):
        self.preclust_eps = value
        
    @pyqtSlot(object)
    def upd_n_guess(self, values):
        self.n_guess = values
        
    @pyqtSlot(bool)
    def upd_n_guess_choice(self, value):
        self.should_use_n_guess = value
        
    @pyqtSlot(float)
    def upd_lambda_exc(self, value):
        self.lambda_exc = value
        
    @pyqtSlot(float)
    def upd_lambda_em(self, value):
        self.lambda_em = value
        
    @pyqtSlot(float)
    def upd_n_s(self, value):
        self.n_s = value
        
    @pyqtSlot(float)
    def upd_n_i(self, value):
        self.n_i = value
        
    @pyqtSlot()
    def upd_NA(self):
        """
        This function reads the correct file, depending on the NA and emission wavelength chosen on UI,
        containing the values of the collection efficiency with respect to z
        """
        match self._view.ui.NA_combobox.currentText():
            case '1.40':
                coll_fl_tab = np.loadtxt(r"src\picasimpler\resources\DF_NA140.txt")
            case '1.42':
                coll_fl_tab = np.loadtxt(r"src\picasimpler\resources\DF_NA142.txt")
            case '1.45':
                coll_fl_tab = np.loadtxt(r"src\picasimpler\resources\DF_NA145.txt")
            case '1.49':
                coll_fl_tab = np.loadtxt(r"src\picasimpler\resources\DF_NA149.txt")
        self.signals.send_coll_fl_tab_toanalysis.emit(coll_fl_tab)
        
    @pyqtSlot(str)
    def _on_calib_done(self, mode=''):
        """
        This function is called upon successfull SIMPLER calibration.
        It saves results on file, both parameters with errors and plot
        """
        if mode=='from file':
            filename_base = self.clust_res_filepath.stem
        else:
            filename_base = self.data_path.stem
        self._print_to_ui(MessageType.INFO, "SIMPLER calibration performed. Results:")
        self._print_to_ui(MessageType.SIMPLE, f"&alpha;<sub>F</sub> = {self._analysis_worker.fit.alpha_F:.3g} &plusmn; {self._analysis_worker.fit.alpha_F_err:.3g}")
        self._print_to_ui(MessageType.SIMPLE, f"d<sub>F</sub> = {self._analysis_worker.fit.d_F:.4g} &plusmn; {self._analysis_worker.fit.d_F_err:.4g} nm")
        self._print_to_ui(MessageType.SIMPLE, f"d<sub>EXC</sub> = {self._analysis_worker.fit.d_exc:.4g} &plusmn; {self._analysis_worker.fit.d_exc_err:.4g} nm")
        self._print_to_ui(MessageType.SIMPLE, f"&theta;<sub>TIRF</sub> = {self._analysis_worker.fit.tirf_angle:.4g}°")
        self._print_to_ui(MessageType.SIMPLE, f"&lt;N<sub>0</sub>&gt; = {self._analysis_worker.fit.N_0_avg:.6g} &plusmn; {self._analysis_worker.fit.N_0_std:.6g}")
        self.save_calib_res(
            filename_base + "_calib_res.json",
            self._analysis_worker.fit.alpha_F,
            self._analysis_worker.fit.alpha_F_err,
            self._analysis_worker.fit.d_F,
            self._analysis_worker.fit.d_F_err,
            self._analysis_worker.fit.d_exc,
            self._analysis_worker.fit.d_exc_err,
            self._analysis_worker.fit.tirf_angle,
            self._analysis_worker.fit.N_0_avg,
            self._analysis_worker.fit.N_0_std
        )
        self.save_calib_plot(filename_base + "_calib_res.png")
        self.save_tirf_angle_plot(filename_base + "_TIRF_angle_plot.png")

    def save_calib_res(self, calib_res_filename, alpha_F, alpha_F_err, d_F, d_F_err, d_exc, d_exc_err, tirf_angle, N_0_avg, N_0_std):
        """
        This function saves the results of the SIMPLER calibration in a .json in the result folder
        """
        calib_res_dict = {
            "alpha_F": alpha_F,
            "alpha_F_err": alpha_F_err,
            "d_F": d_F,
            "d_F_err": d_F_err,
            "d_exc": d_exc,
            "d_exc_err": d_exc_err,
            "TIRF angle": tirf_angle,
            "N_0_avg": N_0_avg,
            "N_0_std": N_0_std
        }
        with open(self.res_dir / Path(calib_res_filename), "w") as f:
            json.dump(calib_res_dict, f, indent=4)
        
    def save_calib_plot(self, calib_plot_filename):
        """
        This function saves the plot of the SIMPLER calibration as a .png in the result folder
        """
        plt.close()
        plt.plot(self._analysis_worker.fit.z_real.ravel(), self._analysis_worker.fit.N_renorm_arr.ravel(), ".", ms=8, label="Data")
        plt.plot(self._analysis_worker.fit.z_ax_forplot, self._analysis_worker.fit.fit_func_forplot.ravel(), label="Fit")
        plt.ylabel("Relative Intensity")
        plt.xlabel("z")
        plt.legend()
        plt.grid()
        plt.savefig(self.res_dir / Path(calib_plot_filename))
        
    def save_tirf_angle_plot(self, tirf_angle_plotname):
        """
        This function saves the plot of the various functions needed to backcalculate the TIRF angle
        """
        plt.close()
        plt.plot(Z_SIM_FIT_ARR, self._analysis_worker.fit.params.coll_fl_interp, label='CF interpolation')
        plt.plot(Z_SIM_FIT_ARR, self._analysis_worker.fit.simpler_prof, label='SIMPLER profile')
        plt.plot(Z_SIM_FIT_ARR, self._analysis_worker.fit.exc_prof, label='excitation profile')
        plt.plot(Z_SIM_FIT_ARR, self._analysis_worker.fit.exc_fit, label='excitation fit')
        plt.legend()
        plt.savefig(self.res_dir / Path(tirf_angle_plotname))
        