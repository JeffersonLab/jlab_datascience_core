from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Residual analyzer:
register(
    id="ResidualAnalyzer_v0",
    entry_point="jlab_datascience_toolkit.analyses.residual_analyzer:ResidualAnalyzer"
)

# Data reconstruction:
register(
    id="DataReconstruction_v0",
    entry_point="jlab_datascience_toolkit.analyses.data_reconstruction:DataReconstruction"
)

# Learning Curve Visualizer:
register(
    id="LearningCurveVisualizer_v0",
    entry_point="jlab_datascience_toolkit.analyses.learning_curve_visualizer:LearningCurveVisualizer"
)
