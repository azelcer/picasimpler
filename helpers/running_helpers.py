#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 17:07:09 2026

@author: azelcer
"""
import logging as _lgn
from threading import Thread as _Thread


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class background_runner:

    # _task_finished_evt = Event()

    def __init__(self):
        self._running = False
        self._thread = None
        self._callback: callable = None

    def submit(self, callback: callable, function: callable, args: list = [], kwargs: dict = {}):
        if self._running or self._thread:
            _lgr.error("Background task already running")
            return False
        self._target = function
        self._callback = callback
        self._thread = _Thread(target=self._do_run, args=args, kwargs=kwargs)
        self._running = True
        self._thread.start()

    def _do_run(self, *args, **kwargs):
        try:
            self._rv = self._target(*args, **kwargs)
        except Exception as e:
            print("exception", e, type(e))
            self._rv = None
        self._running = False
        self._callback(self._rv)

    def cleanup(self):
        if self._running:
            _lgr.error("Background task still running")
            return False
        if not self._thread:
            _lgr.error("No background task running")
            return True
        self._thread.join()
        self._thread = None
        return True
