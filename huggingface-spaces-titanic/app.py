import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(pclass, sex, age, sibsp, parch, fare, cabin):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(fare)
    input_list.append(cabin)

    res = model.predict(np.asarray(input_list).reshape(1, -1))

    if (res[0] == 1):
        return "Passenger survived!"
    else:
        return "Passenger did not survive."


demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Survival Predictive Analysis",
    description="Experiment with the variables below to determine if a passenger would survive the Titanic",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(
            default=1, label="Ticket class (1: upper, 2: middle, 3: lower)"),
        gr.inputs.Number(default=1, label="Gender (1: female, 0: male)"),
        gr.inputs.Number(
            default=1, label="Age (0: child, 1: teenager, 2: young adult, 3: adult, 4: old)"),
        gr.inputs.Number(default=1, label="Siblings/spouse onboard"),
        gr.inputs.Number(default=1, label="Parents/children onboard"),
        gr.inputs.Number(default=1, label="Passenger fare ($)"),
        gr.inputs.Number(
            default=1, label="Cabin deck (0-7 corresponds to A-G)")
    ],

    outputs=gr.Textbox())

demo.launch()
