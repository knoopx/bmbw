from bayes_opt import Events
from bayes_opt.observer import _Tracker
import streamlit as st


class Logger(_Tracker):
    def __init__(self):
        super(Logger, self).__init__()

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])
            st.json(data, expanded=False)
        self._update_tracker(event, instance)
