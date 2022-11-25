import os
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(
        ["hopsworks", "joblib", "seaborn", "sklearn", "dataframe-image"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def cabin_encoder(cabin):
    list_of_decks = ["A", "B", "C", "D", "E", "F", "T", "G"]
    for deckNumber in list_of_decks:
        if deckNumber in cabin:
            return ord(deckNumber)
    # -1 means unknown cabin number (unknown deck)
    return -1


def age_encoder(age):
    #Child = 0
    if 0 <= age <= 13:
        return 0
    #Teenager = 1
    if 12 <= age <= 19:
        return 1
    # Young adult = 2
    if 20 <= age <= 35:
        return 2
    #Adult = 3
    if 36 <= age <= 59:
        return 3
    #Old = 4
    if 60 <= age:
        return 4
    # If age is not recorded
    else:
        return -1


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

    # Preprocessing --------------------------------------------------------------------------------
    # Dropping four features that are unlikely to affect the result:
    #  Name: the name of the passenger, each one is unique
    #  PassengerId: the ID of the passenger, each one is unique
    #  Ticket: just the ID number of the ticket, each one is unique
    #  Embarked: the port from which you embarked
    titanic_df = titanic_df.drop(
        columns=["Name", "PassengerId", "Ticket", "Embarked"])

    # Encode categorical data (sex and cabin)
    titanic_df["Cabin"] = titanic_df["Cabin"].map(
        lambda x: cabin_encoder(str(x)))
    titanic_df["Sex"] = titanic_df["Sex"].apply(
        lambda x: 1 if x == "female" else 0)

    # Group age groups together for easier analysis and fill missing values
    titanic_df["Age"] = titanic_df["Age"].map(
        lambda x: age_encoder(x))

    # Specify type on remaining features and fill potential missing values
    titanic_df["Survived"] = titanic_df["Survived"].fillna(-1)
    titanic_df["Pclass"] = titanic_df["Pclass"].fillna(-1)
    titanic_df["SibSp"] = titanic_df["SibSp"].fillna(-1)
    titanic_df["Parch"] = titanic_df["Parch"].fillna(-1)
    titanic_df["Fare"] = titanic_df["Fare"].fillna(-1)

    # Feature group --------------------------------------------------------------------------------
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal",
        version=1,
        primary_key=["PassengerId", "Pclass", "Name", "Sex", "Age",
                     "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        description="Titanic passengers dataset")

    titanic_fg.insert(titanic_df, write_options={"wait_for_job": False})

    print(titanic_df.info())
    print(titanic_df.head(10))


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
