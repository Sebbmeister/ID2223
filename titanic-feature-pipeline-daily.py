import os
import modal

BACKFILL = False
LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "sklearn", "dataframe-image"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def generate_synthetic_passenger():
    import pandas as pd
    import random

    passenger_df = pd.DataFrame(
        {
            "Pclass": [random.choice([1, 2, 3])],
            "Sex": [random.choice([0, 1])],
            "Age": [random.choice([0, 1, 2, 3, 4])],
            "SibSp": [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
            "Parch": [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
            "Fare": [random.uniform(0, 250)],
            "Cabin": [int(random.choice([0, 1, 2, 3, 4, 5, 6, 7]))]
        }
    )
    passenger_df["Survived"] = random.choice([0, 1])
    return passenger_df


def get_synthetic_passenger():
    import pandas as pd
    import random

    synthetic_passenger = generate_synthetic_passenger()
    return synthetic_passenger


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv(
            "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    else:
        titanic_df = get_synthetic_passenger()

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["PassengerId", "Pclass", "Name", "Sex", "Age",
                     "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        description="Titanic passengers dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
