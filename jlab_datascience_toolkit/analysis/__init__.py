from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

# Residual analyzer:
register(
    id="ResidualAnalyzer_v0",
    entry_point="jlab_datascience_toolkit.analysis.residual_analyzer:ResidualAnalyzer"
)

from jlab_datascience_toolkit.analysis.residual_analyzer import ResidualAnalyzer
