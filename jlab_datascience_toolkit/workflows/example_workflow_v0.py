import yaml
import argparse
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from jlab_datascience_toolkit.data_prep import make as make_prep
from jlab_datascience_toolkit.models import make as make_model
from jlab_datascience_toolkit.trainers import make as make_trainer
from jlab_datascience_toolkit.analysis import make as make_analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="../cfgs/defaults/multiclass_cfg.yaml", help='Path to yaml configuration file')
    parser.add_argument("--logdir", type=str, default="./", help='Path logging directory. If None, analysis figures will not be saved.')
    args = parser.parse_args()
    args = vars(args)     # convert args from argparse.Namespace to dict

    with open(args['cfg_file'], 'r') as file:
        configs = yaml.safe_load(file)
        prep_configs = configs['prep_configs']
        model_configs = configs['model_configs']
        trainer_configs = configs['trainer_configs']
        analysis_configs = configs['analysis_configs']

    # 1) Load Data
    df = sns.load_dataset('iris')
    classes_list = [(c, i) for i, c in enumerate(df['species'].unique().tolist())]
    df['species_int'] = df['species'].map(dict(classes_list))
    
    # 2) Split Data
    prep = make_prep(prep_configs['registered_name'], configs=prep_configs)
    x_train, x_val, x_test, y_train, y_val, y_test = prep.run(df)
    
    # 3) Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # 4) Define Model
    model = make_model(model_configs['registered_name'], configs=model_configs)
    
    # 5) Train Model
    trainer = make_trainer(trainer_configs['registered_name'], configs=trainer_configs)
    history = trainer.fit(model=model, x=x_train, y=y_train, validation_data=(x_val, y_val))

    # 6) Analyze Model on test dataset
    y_pred = model.predict(x_test)    # (n_samples, c_classes)
    y_pred = y_pred.argmax(axis=1)    # (n_samples)
    multiclass_ana = make_analysis(analysis_configs["registered_name"], configs=analysis_configs)
    results = multiclass_ana.run(
        y_test,
        y_pred,
        labels = [tup[1] for tup in classes_list],
        target_names = [tup[0] for tup in classes_list],
        logdir = args['logdir']
    )
    print(results)