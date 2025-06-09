import streamlit as st
from fastai.vision.all import *
from pathlib import Path
from PIL import Image
import os

# --- กำหนด Path ไปยังไฟล์โมเดล ---
MODEL_PATH = Path("convnextv2_thev1_best_for_good.pkl")  # แก้ไข path ของไฟล์โมเดลให้ตรง

# --- ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_model(model_path):
    try:
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงหรือไม่
        if not model_path.is_file():
            st.error(f"Model file NOT FOUND at {model_path}. Please check the path.")
            return None
        
        # โหลดโมเดลด้วย fastai
        learn = load_learner(model_path)
        st.success("Model loaded successfully!")
        return learn
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check the model and dependencies.")
        return None

# --- โหลดโมเดล ---
learn = load_model(MODEL_PATH)

# --- ส่วนแสดงผล Streamlit ---
st.title("💩 Poop Classification")
st.subheader("Upload your Poop Image")

# --- ส่วนอัปโหลดไฟล์ ---
uploaded_file = st.file_uploader("Click to upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_to_display = Image.open(uploaded_file)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("##### This is your Image")
        st.image(image_to_display, use_column_width=True)

        if st.button("🚽 Click For Predict"):
            with st.spinner("Predicting..."):
                img_bytes = uploaded_file.getvalue()

                try:
                    pil_image = PILImage.create(img_bytes)  # แปลงไฟล์ที่อัปโหลดเป็น PILImage
                except Exception as e_pil:
                    st.error(f"Error converting uploaded file to image: {e_pil}")
                    st.stop()  # หยุดการทำงานหากเกิดข้อผิดพลาด

                try:
                    # ทำการทำนายจากโมเดล
                    pred_class, pred_idx, probs = learn.predict(pil_image)
                    st.markdown(f"#### Result is: **{pred_class}**")
                    st.markdown(f"##### Probability: **{probs[pred_idx]:.1%}**")  # แสดงผลเปอร์เซ็นต์ของการทำนาย

                    # เก็บข้อมูลการทำนายใน session state
                    st.session_state.prediction_made = True
                    st.session_state.predicted_class = pred_class
                    st.session_state.probabilities = probs.numpy()
                    st.session_state.class_names = list(learn.dls.vocab)
                except Exception as e_predict:
                    st.error(f"Error during prediction: {e_predict}")

    # --- แสดงผลการทำนาย ---
    if 'prediction_made' in st.session_state and st.session_state.prediction_made:
        with col2:
            st.markdown("##### Probabilities Chart")
            # สร้าง DataFrame สำหรับแสดงกราฟ
            df_probs = pd.DataFrame({
                'Class': st.session_state.class_names,
                'Probability': st.session_state.probabilities * 100
            })
            # ใช้ plotly สำหรับแสดงกราฟ pie chart
            import plotly.express as px
            fig = px.pie(df_probs, values='Probability', names='Class', title='Prediction Probabilities')
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("App by Chokun7788 (with AI assistant)")
