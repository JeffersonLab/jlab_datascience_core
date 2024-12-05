import yaml
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.preprocessing import StandardScaler
from jlab_datascience_toolkit.data_parsers import make as make_parser
from jlab_datascience_toolkit.data_preps import make as make_prep
from jlab_datascience_toolkit.models import make as make_model
from jlab_datascience_toolkit.trainers import make as make_trainer
from jlab_datascience_toolkit.analyses import make as make_analysis

@hydra.main(version_base=None, config_path="../cfgs/defaults", config_name="multiclass_cfg")
def main(configs: DictConfig):
    configs = OmegaConf.to_container(configs)    # convert DictConfig ==> dict
    logdir = configs["logdir"]
    parser_configs = configs["parser_configs"]
    prep_configs = configs["prep_configs"]
    model_configs = configs["model_configs"]
    trainer_configs = configs["trainer_configs"]
    analysis_configs = configs["analysis_configs"]

    # 1) Load Data
    parser = make_parser(parser_configs["registered_name"], configs=parser_configs)
    df = parser.load_data()
    classes_list = [(c, i) for i, c in enumerate(df["species"].unique().tolist())]
    df["species_int"] = df["species"].map(dict(classes_list))

    # 2) Split Data
    prep = make_prep(prep_configs["registered_name"], configs=prep_configs)
    x_train, x_val, x_test, y_train, y_val, y_test = prep.run(df)

    # 3) Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # 4) Define Model
    model = make_model(model_configs["registered_name"], configs=model_configs)

    # 5) Train Model
    trainer = make_trainer(trainer_configs["registered_name"], configs=trainer_configs)
    history = trainer.fit(
        model=model, x=x_train, y=y_train, validation_data=(x_val, y_val), logdir=logdir
    )

    # 6) Analyze Model on test dataset
    y_pred = model.predict(x_test)  # (n_samples, c_classes)
    y_pred = y_pred.argmax(axis=1)  # (n_samples)
    multiclass_ana = make_analysis(
        analysis_configs["registered_name"], configs=analysis_configs
    )
    results = multiclass_ana.run(
        y_test,
        y_pred,
        labels=[tup[1] for tup in classes_list],
        target_names=[tup[0] for tup in classes_list],
        logdir=logdir,
    )
    print(results)


if __name__ == "__main__":
    main()
