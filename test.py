import streamlit as st
from fastai.vision.all import *
from pathlib import Path
import pathlib # สำหรับ PosixPath patch
import sys
from PIL import Image # Import Image จาก Pillow โดยตรงด้วยก็ดีครับ (แม้ fastai.vision.all จะมี PILImage)
import pandas as pd
import plotly.express as px

# --- Monkey patch สำหรับ PosixPath บน Windows (สำคัญถ้าโมเดล export จาก Colab/Linux) ---
_original_posix_path = None 
# if sys.platform == "win32":
#     if hasattr(pathlib, 'PosixPath') and not isinstance(pathlib.PosixPath, pathlib.WindowsPath):
#         _original_posix_path = pathlib.PosixPath
#         pathlib.PosixPath = pathlib.WindowsPath
#         print("INFO: PosixPath patch applied for Streamlit app.")
# --- สิ้นสุดส่วนโค้ด Patch ---

# --- !!! สำคัญมาก: แก้ไข Path ไปยังไฟล์โมเดล .pkl ของคุณที่นี่ !!! ---
MODEL_PATH = Path("convnextv2_thev1_best_for_good.pkl") # <--- ***** แก้ไข Path นี้ *****
# หรือ MODEL_PATH = Path("Thisisthelastone.pkl") # ถ้าไฟล์อยู่ในโฟลเดอร์เดียวกับ app_streamlit.py
# --------------------------------------------------------------------

@st.cache_resource 
def load_my_model(model_path):
    print(f"Attempting to load model from: {model_path}")
    try:
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงหรือไม่ก่อนโหลด
        if not model_path.is_file():
            st.error(f"Model file NOT FOUND at {model_path}. Please check the path in your Streamlit script.")
            return None
        learn = load_learner(model_path)
        print("Model loaded successfully!")
        return learn
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure the model file is valid and all dependencies are installed (e.g., timm, cloudpickle, fasttransform if needed by your model).")
        return None

# โหลดโมเดล
learn = load_my_model(MODEL_PATH)

# --- ส่วนของหน้าตา Streamlit App ---
st.title("💩 Poop Classification")

st.subheader("Upload your Poop Image")
uploaded_file = st.file_uploader(
    "Click This For Upload Image", 
    type=["jpg", "jpeg", "png"], 
    help="Limit 200MB per file"
)

# if learn is None:
#     st.error("AI Model could not be loaded. Please check the server logs or model path configuration.")
#     st.stop() # หยุดการทำงานของ app ถ้าโมเดลโหลดไม่ได้

if uploaded_file is not None:
    image_to_display = Image.open(uploaded_file)
    
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("##### This is your Image")
        st.image(image_to_display, use_column_width=True)

        if st.button("🚽 Click For Predict", use_container_width=True):
            with st.spinner("Predicting..."):
                img_bytes = uploaded_file.getvalue()
                
                # --- !!! ส่วนที่แก้ไข: แปลง bytes เป็น PILImage ก่อน predict !!! ---
                try:
                    pil_image = PILImage.create(img_bytes) # ใช้ PILImage.create ของ FastAI
                except Exception as e_pil:
                    st.error(f"Error converting uploaded file to an image: {e_pil}")
                    # ถ้าแปลงเป็น PILImage ไม่ได้ ให้เคลียร์ session state ที่เกี่ยวกับการ predict (ถ้ามี)
                    if 'prediction_made' in st.session_state:
                        del st.session_state.prediction_made 
                    st.stop() # หยุดการทำงานส่วนนี้ถ้าแปลงรูปไม่ได้
                # --- สิ้นสุดส่วนที่แก้ไข ---

                try:
                    # ส่ง PILImage object เข้าไปใน learn.predict()
                    pred_class, pred_idx, probs = learn.predict(pil_image) 
                    
                    st.markdown("---")
                    st.markdown(f"#### Result is : **{pred_class}**")
                    st.markdown(f"##### {pred_class} : **{probs[pred_idx]:.1%}**") # แสดงทศนิยม 1 ตำแหน่งเปอร์เซ็นต์
                
                    st.session_state.prediction_made = True
                    st.session_state.predicted_class = pred_class
                    st.session_state.probabilities = probs.numpy() 
                    st.session_state.class_names = list(learn.dls.vocab)
                except Exception as e_predict:
                    st.error(f"Error during prediction: {e_predict}")
                    if 'prediction_made' in st.session_state:
                        del st.session_state.prediction_made
    
    if 'prediction_made' in st.session_state and st.session_state.prediction_made:
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Probabilities Chart")
            
            df_probs = pd.DataFrame({
                'Class': st.session_state.class_names,
                'Probability': st.session_state.probabilities * 100
            })
            
            fig = px.pie(df_probs, values='Probability', names='Class', 
                         title='Prediction Probabilities',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+name')
            fig.update_layout(showlegend=True, legend_title_text='Classes') # แสดง Legend พร้อมชื่อ
            
            st.plotly_chart(fig, use_container_width=True)
        # เคลียร์ session state หลังแสดงผล เพื่อให้พร้อมสำหรับการ predict ครั้งถัดไป (ถ้าต้องการ)
        # หรือจะปล่อยไว้เพื่อให้ผลลัพธ์ล่าสุดยังคงแสดงอยู่ก็ได้
        # del st.session_state.prediction_made 

st.markdown("---")
st.caption("App by Chokun7788 (with AI assistant)")
