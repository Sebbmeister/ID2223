import os
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    hopsworks_image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "sklearn", "dataframe-image"])

    @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    survived = y_pred[y_pred.size-1]

    #Get survived/died images + dataset api
    dataset_api = project.get_dataset_api()
    sad_img_url = "https://raw.githubusercontent.com/Sebbmeister/ID2223/main/sad.png"
    happy_img_url = "https://raw.githubusercontent.com/Sebbmeister/ID2223/main/smiley.png"
    sad_img = Image.open(requests.get(sad_img_url, stream=True).raw)
    happy_img = Image.open(requests.get(happy_img_url, stream=True).raw)

    if (survived == 1):
        print("The passenger is predicted to have survived!")
        happy_img.save("./latest_titanic_prediction.png")
        dataset_api.upload("./latest_titanic_prediction.png", "Resources/images", overwrite=True)
    else:
        print("The passenger is predicted to have died :(")
        sad_img.save("./latest_titanic_prediction.png")
        dataset_api.upload("./latest_titanic_prediction.png", "Resources/images", overwrite=True)

    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read()

    label = df.iloc[-1]["survived"]
    if (label == 1):
        survived_status = "survived"
        happy_img.save("./latest_titanic_actual.png")
        dataset_api.upload("./latest_titanic_actual.png", "Resources/images", overwrite=True)
    else:
        survived_status = "died"
        sad_img.save("./latest_titanic_actual.png")
        dataset_api.upload("./latest_titanic_actual.png", "Resources/images", overwrite=True)
    print("The passenger " + survived_status)

    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Survival Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survived],
        'label': [label],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(5)
    dfi.export(df_recent, './titanic_df_recent.png', table_conversion='matplotlib')
    dataset_api.upload("./titanic_df_recent.png", "Resources/images", overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)

        df_cm = pd.DataFrame(results, ['True Died', 'True Survived'],
                             ['Pred Died', 'Pred Survived'])

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./titanic_confusion_matrix.png")
        dataset_api.upload("./titanic_confusion_matrix.png",
                           "Resources/images", overwrite=True)

    else:
        print("Two predictions needed, current number of predictions: ",
              predictions.value_counts().count())


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
