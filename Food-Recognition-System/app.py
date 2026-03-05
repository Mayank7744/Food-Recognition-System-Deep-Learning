import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("model/food_recognition_model.h5")

class_names = [
"burger","butter_naan","chai","chapati","chole_bhature",
"dal_makhani","dhokla","fried_rice","idli","jalebi",
"kaathi_rolls","kadai_paneer","kulfi","masala_dosa",
"momos","paani_puri","pakode","pav_bhaji","pizza","samosa"
]

def predict(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)[0]

    result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Food Recognition System",
    description="Upload an image of food to detect its category."
)

demo.launch()