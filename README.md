### Indoor navigation for drone sharing space with people ###

## part  1 - realtime processing ##

Located in folder ``real-time``.

This whole part rely on an event camera from IniVation and their python library ``dv_processing`` (see https://gitlab.com/inivation/dv/dv-processing/) to work properly.


- `` dv_calibration.py`` is used to convert a calibration file from DV into usable information.
-  ``record.py`` is used to record a stream of events for later processing.
- ``camshift.py``, ``contour.py``and ``tracking.py`` perform real time tracking using different methods.


## part 2 - experiment post-processing ##

Located in folder ``data``.

Runs a comparison between ground-truth and position estimation for the experiment designed to test the proposed methods for this internship.

Comparison done by running ``contour_comparator.py`` or ``featour_comparator.py``. Each script compares a single method (contour or feature) against the ground-truth gathered during the experiment.
