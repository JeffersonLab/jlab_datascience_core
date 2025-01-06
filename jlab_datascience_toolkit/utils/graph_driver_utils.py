from typing import NamedTuple, Union, Iterable
from jlab_datascience_toolkit.utils.registration import make 
    
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
    
    def get_module_dict(self, modules, config_kwargs_list):
        module_dict = dict.fromkeys(modules, None)
        for m_name in module_dict:
            module_id = modules[m_name]
            print(f'Making {m_name} with module ID: {module_id}')
            config_kwargs = config_kwargs_list[m_name]
            module_dict[m_name] = make(module_id, **config_kwargs)

        return module_dict
    
    def get_args(self, data, module_dict, input):

        split_input = input.split(sep=".")
        if split_input[0] == "module":
            data_in = module_dict[split_input[1]]
        elif len(split_input) == 1:
            data_in = data[input]
        else:
            raise KeyError(f"No input type: {split_input[0]}.")

        return data_in

    def convert_graph_lists_to_tuples(self, graph):
        for edge in graph:
            if isinstance(edge[0], list):
                edge[0] = tuple(edge[0])
            if isinstance(edge[2], list):
                edge[2] = tuple(edge[2])

    def run_graph(self, graph, modules, config_kwargs_list):
        self.convert_graph_lists_to_tuples(graph)
        graph_edges = self.tuples_to_edges(graph)
        data = self.get_distinct_data_dict(graph_edges)
        module_dict = self.get_module_dict(modules, config_kwargs_list)
        for edge in graph_edges:
        
            if '.' in edge.function:
                m_name, fn_call = edge.function.split('.')
                fn = getattr(module_dict[m_name], fn_call)
            else:
                fn = getattr(self, edge.function)

            if edge.input is None:
                fn_in = [] #Unpacks to 0 arguments
            elif isinstance(edge.input, str):
                data_in = self.get_args(data, module_dict, edge.input)
                fn_in = [data_in] # Unpacks to 1 argument
            elif isinstance(edge.input, Iterable):
                fn_in = [self.get_args(data, module_dict, val) for val in edge.input]
            
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

    def print(self, input):
        print(input)
