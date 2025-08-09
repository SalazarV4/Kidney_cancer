import gradio as gr
from kidney_cancer.pipeline.prediction import PredictionPipeline


gr.Interface(fn=PredictionPipeline.predict,
             inputs=gr.Image(type='pil'),
             outputs=gr.Label(num_top_classes=2),
             examples=["artifacts/data_ingestion/Kidney_dataset/Test/Normal/Normal- (449).jpg",
                       "artifacts/data_ingestion/Kidney_dataset/Test/Tumor/Tumor- (450).jpg"]).launch()
