from typing import NamedTuple, Union, Iterable
from jlab_datascience_toolkit.utils.registration import make 
import jlab_datascience_toolkit.data_parser
import jlab_datascience_toolkit.data_prep
import jlab_datascience_toolkit.model
import jlab_datascience_toolkit.analysis

'''
This class was developed by Steven Goldenberg (sgolden@jlab.org) and helps set up a generic driver in an elegant way.
'''
    
class GraphRuntime():
    class Edge(NamedTuple):
        input: Union[str, tuple]
        function: str
        output: Union[str, tuple]

    def tuples_to_edges(self,tuple_list):
        edges = []
        for tuple in tuple_list:
            edges.append(self.Edge(*tuple))

        return edges
    
    def get_distinct_data_dict(self, graph_edges):
        distinct_data = set()
        for edge in graph_edges:
            if edge.input is not None:
                if isinstance(edge.input, str):
                    distinct_data.add(edge.input)
                else:
                    [distinct_data.add(val) for val in edge.input]
            if edge.output is not None:
                if isinstance(edge.output, str):
                    distinct_data.add(edge.output)
                else:
                    [distinct_data.add(val) for val in edge.output]

        return dict.fromkeys(distinct_data, None)
    
    def get_module_dict(self,modules,config_paths,user_configs):
        module_dict = dict.fromkeys(modules, None)
        for m_name in module_dict:
            module_id = modules[m_name]
            print(f'Making {m_name} with module ID: {module_id}')
            module_dict[m_name] = make(module_id,path_to_cfg=config_paths[m_name],user_config=user_configs[m_name])

        return module_dict
    
    def run_graph(self, graph, modules,config_paths,user_configs):
        graph_edges = self.tuples_to_edges(graph)
        data = self.get_distinct_data_dict(graph_edges)
        module_dict = self.get_module_dict(modules,config_paths,user_configs)
        for edge in graph_edges:
        
            if '.' in edge.function:
                m_name, fn_call = edge.function.split('.')
                fn = getattr(module_dict[m_name], fn_call)
            else:
                fn = getattr(self, edge.function)

            if edge.input is None:
                fn_in = [] #Unpacks to 0 arguments
            elif isinstance(edge.input, str):
                fn_in = [data[edge.input]] # Unpacks to 1 argument
            elif isinstance(edge.input, Iterable):
                fn_in = [data[val] for val in edge.input]
            
            # Take advantage of list unpacking for arguments
            out = fn(*fn_in)

            if out is not None:
                if isinstance(edge.output, tuple):
                    for o, d in zip(out, edge.output):
                        data[d] = o
                else:
                    data[edge.output] = out

        return data, module_dict

    def combine(self, *inputs):
        return inputs