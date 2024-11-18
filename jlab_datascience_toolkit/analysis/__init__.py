from jlab_datascience_toolkit.utils.registration import register, make, list_registered_modules

register(
    id="MultiClassClassificationAnalysis_v0",
    entry_point="jlab_datascience_toolkit.analysis.multiclass_analysis_v0:Analysis"
)
