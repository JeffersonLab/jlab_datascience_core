from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Residual analyzer:
register(
    id="ResidualAnalyzer_v0",
    entry_point="jlab_datascience_toolkit.analysis.residual_analyzer:ResidualAnalyzer"
)

from jlab_datascience_toolkit.analysis.residual_analyzer import ResidualAnalyzer

# Data reconstruction:
register(
    id="DataReconstruction_v0",
    entry_point="jlab_datascience_toolkit.analysis.data_reconstruction:DataReconstruction"
)

from jlab_datascience_toolkit.analysis.data_reconstruction import DataReconstruction

# Learning Curve Visualizer:
register(
    id="LearningCurveVisualizer_v0",
    entry_point="jlab_datascience_toolkit.analysis.learning_curve_visualizer:LearningCurveVisualizer"
)

from jlab_datascience_toolkit.analysis.learning_curve_visualizer import LearningCurveVisualizer
