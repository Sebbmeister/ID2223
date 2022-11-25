import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_titanic_prediction.png", overwrite=True)
dataset_api.download("Resources/images/latest_titanic_actual.png", overwrite=True)
dataset_api.download("Resources/images/titanic_df_recent.png", overwrite=True)
dataset_api.download("Resources/images/titanic_confusion_matrix.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Latest Survival Prediction")
          input_img = gr.Image("latest_titanic_prediction.png", elem_id="predicted-img")
      with gr.Column():          
          gr.Label("Latest Survival Actual")
          input_img = gr.Image("latest_titanic_actual.png", elem_id="actual-img")        
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("titanic_df_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("titanic_confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch()
