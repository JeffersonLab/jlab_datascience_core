import jlab_datascience_toolkit.data_parsers as parsers
import jlab_datascience_toolkit.data_preps as preps
import jlab_datascience_toolkit.models as models
import jlab_datascience_toolkit.agents as agents


modules = {
    'data parsers': parsers,
    'data preps': preps,
    'models': models,
    'agents': agents
}

print(" ")
for mod in modules:
    print(f"Available {mod}:")
    print(modules[mod].list_registered_modules())
    print(" ")