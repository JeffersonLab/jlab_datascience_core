import unittest
import jlab_datascience_toolkit.data_parsers as parsers
import jlab_datascience_toolkit.data_preps as preps
import jlab_datascience_toolkit.models as models
import jlab_datascience_toolkit.agents as agents
import yaml

class TestRegistry(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRegistry, self).__init__(*args, **kwargs)
        self.modules = {
         'data parsers': parsers,
         'data preps': preps,
         'models': models,
         'agents': agents
        }
    
    def test_registry(self):
        for mod in self.modules:
            available_modules = self.modules[mod].list_registered_modules()

            if len(available_modules) > 0:
              for id in available_modules:
                print(f"Making module: {id}...")
                self.modules[mod].make(id,config=None)
                print("...done!")
                print(" ")
            else:
                print(f"Not modules registered for: {mod}")

if __name__ == "__main__":
    unittest.main()
