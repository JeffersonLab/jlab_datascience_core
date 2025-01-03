import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.preprocessing import StandardScaler
import jlab_datascience_toolkit
from jlab_datascience_toolkit.utils.graph_driver_utils import GraphRuntime


class CustomGraphRuntime(GraphRuntime):
    def map_species(self, df):
        classes_list = [(c, i) for i, c in enumerate(df["species"].unique().tolist())]
        df["species_int"] = df["species"].map(dict(classes_list))
        labels=[tup[1] for tup in classes_list]
        target_names=[tup[0] for tup in classes_list]
        return df, labels, target_names

    def argmax(self, data):
        return data.argmax(axis=1)

from jlab_datascience_toolkit.utils.registration import register
register(id="SKLearnStandardScaler", entry_point=StandardScaler)

@hydra.main(version_base=None, config_path="../cfgs/defaults", config_name="multiclass_graph_cfg")
def main(configs: DictConfig):

    configs = OmegaConf.to_container(configs)    # convert DictConfig ==> dict

    graph = configs["graph"]
    modules = configs["modules"]
    config_kwargs_list = configs["kwargs_list"]

    graph_runtime = CustomGraphRuntime()
    data, module_dict = graph_runtime.run_graph(graph=graph, modules=modules, config_kwargs_list=config_kwargs_list)

    # logdir = configs.get("logdir", None)
    # if logdir is None:
    #     logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

if __name__ == "__main__":
    main()
