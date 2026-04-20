import streamlit as st
from PIL import Image
import torch
import numpy as np
from model import CycleGAN
from transforms import val_transform_a, val_transform_b, de_normalize_a, de_normalize_b

# torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])

device = "cuda" if torch.cuda.is_available() else "cpu"

if "model" not in st.session_state:
    model = CycleGAN()
    state_dict = torch.load("weights.pt", map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    st.session_state.model = model

def preprocess_image(image: Image.Image, transform):
    tensor = transform(image)
    return tensor.unsqueeze(0).to(device)

def tensor_to_image(tensor, de_normalize):
    img_tensor = tensor.squeeze(0)
    img = de_normalize(img_tensor)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def rgb_to_bgr(img):
    r, g, b = img.split()
    return Image.merge("RGB", (b, g, r))

st.title("Превратите лошадь в зебру и наоборот!")
st.write("Загрузите картинку и выберите целевой класс (Лошадь→Зебра или Зебра→Лошадь).")

target_class = st.radio("Класс входного изображения:", ("Лошадь", "Зебра"), horizontal=True)

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Оригинал", use_container_width=True)

    if st.button("Преобразовать"):
        with st.spinner("Обработка..."):
            model = st.session_state.model

            if target_class == "Лошадь":
                input_tensor = preprocess_image(rgb_to_bgr(original_image), val_transform_a)
                output_tensor = model.get_fake_b(input_tensor)
                print(output_tensor.shape)
                result_image = tensor_to_image(output_tensor, de_normalize_b)
                caption = "Ваша зебра"
            else:
                input_tensor = preprocess_image(rgb_to_bgr(original_image), val_transform_b)
                output_tensor = model.get_fake_a(input_tensor)
                result_image = tensor_to_image(output_tensor, de_normalize_a)
                caption = "Ваша лошадь"

            st.image(result_image, caption=caption, use_container_width=True)
else:
    st.info("Пожалуйста, загрузите изображение.")
