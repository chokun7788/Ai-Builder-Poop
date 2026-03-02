import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import pathlib
import google.generativeai as genai 

# --- 1. จัดการ Path สำหรับ Windows/Linux (Compatibility) ---
if sys.platform == "win32":
    if hasattr(pathlib, 'PosixPath') and not isinstance(pathlib.PosixPath, pathlib.WindowsPath):
        pathlib.PosixPath = pathlib.WindowsPath

# --- 2. การตั้งค่า Gemini API และระบบ Debug ---
api_key_configured = False
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # ตรวจสอบว่า Key นี้มองเห็นโมเดลรุ่นที่เราต้องการหรือไม่
        available_models = [m.name for m in genai.list_models()]
        if 'models/gemini-1.5-flash' in available_models:
            api_key_configured = True
        else:
            st.error(f"⚠️ API Key นี้มองไม่เห็นรุ่น gemini-1.5-flash รุ่นที่มีคือ: {available_models}")
    else:
        st.error("❌ ไม่พบ GOOGLE_API_KEY ใน Streamlit Secrets (โปรดเช็คใน Settings > Secrets)")
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}")

# --- 3. ฟังก์ชันเรียกใช้งาน Gemini (อธิบายผลเบื้องต้น) ---
def get_initial_explanation(stool_class):
    if not api_key_configured:
        return "ขออภัยครับ ไม่สามารถเชื่อมต่อกับ AI ได้ในขณะนี้ โปรดตรวจสอบการตั้งค่า API Key"

    class_map = {
        "Blood": "มีเลือดปน (Blood)", "Diarrhea": "ท้องร่วง/ท้องเสีย (Diarrhea)",
        "Green": "สีเขียว (Green)", "Mucus": "มีมูกปน (Mucus)",
        "Normal": "ปกติ (Normal)", "Yellow": "สีเหลือง (Yellow)"
    }
    friendly_name = class_map.get(stool_class, stool_class)
    
    prompt = f"""
    ในฐานะผู้เชี่ยวชาญด้านสุขภาพเบื้องต้น โปรดให้ข้อมูลเกี่ยวกับอุจจาระประเภท "{friendly_name}" เพื่อเริ่มต้นการสนทนา
    กรุณาอธิบายโดยละเอียดเป็นภาษาไทย โดยแบ่งหัวข้อให้ชัดเจนดังนี้:
    1. **สาเหตุที่เป็นไปได้:**
    2. **ความเสี่ยงหรือโรคที่อาจเกี่ยวข้อง:**
    3. **คำแนะนำเบื้องต้นและการดูแลตัวเอง:**
    **คำเตือนสำคัญ:** ข้อมูลนี้เป็นเพียงคำแนะนำเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้ 
    และจบด้วยการบอกว่า "หากมีคำถามเพิ่มเติมเกี่ยวกับผลลัพธ์นี้ สามารถพิมพ์ถามได้เลยครับ"
    """
    
    # ใช้ชื่อรุ่นแบบเต็มเพื่อป้องกัน Error NotFound
    model = genai.GenerativeModel('models/gemini-1.5-flash') 
    response = model.generate_content(prompt)
    return response.text

# --- 4. การจัดการโมเดล Fastai ---
MODEL_FILENAME = Path("convnextv2_thev1_best_for_good.pkl")

@st.cache_resource
def load_model(local_path):
    return load_learner(local_path)

# พยายามโหลดโมเดล
try:
    learn = load_model(MODEL_FILENAME)
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดไฟล์โมเดลได้: {e}")
    st.stop()

# --- 5. ส่วนแสดงผล UI ---
st.title("💩 :rainbow[Poop Classification & AI Chat]")
st.subheader("แยกประเภทอุจจาระ และพูดคุยถาม-ตอบกับ AI")
st.warning("⚠️ **ข้อควรระวัง:** ผลลัพธ์จาก AI นี้เป็นเพียงข้อมูลเบื้องต้นเพื่อการศึกษาเท่านั้น **ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้**")

# ฟังก์ชันจัดการการทำนายและเริ่มแชท
def process_and_start_chat(image_source, key_suffix):
    if st.button("ทำนาย และ อธิบาย", key=key_suffix):
        with st.spinner('กำลังประมวลผล...'):
            # ทำนายด้วย Fastai
            pil_image = PILImage.create(image_source)
            pred_class, pred_idx, probs = learn.predict(pil_image)
            
            st.markdown(f"#### ผลลัพธ์: **{pred_class}**")
            st.markdown(f"##### ความน่าจะเป็น: **{probs[pred_idx]:.1%}**")
            
            # กราฟวงกลมแสดงความน่าจะเป็น
            df_probs = pd.DataFrame({'Class': learn.dls.vocab, 'Probability': probs.numpy() * 100})
            fig = px.pie(df_probs, values='Probability', names='Class', color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # เรียก Gemini อธิบายผล
            if api_key_configured:
                with st.spinner('AI กำลังเตรียมคำอธิบาย...'):
                    initial_explanation = get_initial_explanation(pred_class)
                    model = genai.GenerativeModel('models/gemini-1.5-flash')
                    st.session_state.chat = model.start_chat(history=[])
                    st.session_state.messages = [{"role": "model", "parts": [initial_explanation]}]
            else:
                st.error("ไม่สามารถเรียกใช้ AI อธิบายได้ เนื่องจากปัญหา API Key")

# --- 6. เมนูเลือกโหมดใช้งาน ---
sec = st.selectbox("เลือกหมวดหมู่", ["อัปโหลดรูปเพื่อใช้งานจริง", "ทดลองใช้(ตัวอย่างรูป)"])

if sec == "อัปโหลดรูปเพื่อใช้งานจริง":
    upload_file = st.file_uploader("อัปโหลดภาพของคุณ", type=["jpg", "jpeg", "png"])
    if upload_file:
        st.image(upload_file, caption="ภาพที่อัปโหลด", use_container_width=True)
        process_and_start_chat(upload_file, key_suffix="upload")

elif sec == "ทดลองใช้(ตัวอย่างรูป)":
    class_poo = st.selectbox("เลือกคลาสที่ต้องการทดสอบ", ["Blood", "Diarrhea", "Green", "Mucus", "Normal", "Yellow"])
    
    # Path รูปตัวอย่าง (ต้องตรงกับใน GitHub)
    ex_img = {
        "Blood": ["Image/Blood/1.png", "Image/Blood/2.jpg"],
        "Diarrhea": ["Image/Diarrhea/68621499_10158897636364968_929960603991146496_n.jpg"],
        "Green": ["Image/Green/470220113_1111404843690107_5400214401912539739_n.jpg"],
        "Mucus": ["Image/Mucus/does-this-look-like-it-could-be-worms-or-maybe-mucus-in-my-v0-6fvtr2ywdv4d1.png"],
        "Normal": ["Image/Normal/54.png"],
        "Yellow": ["Image/Yellow/470467721_122113376150620788_7483223442733841889_n.jpg"]
    }

    if class_poo in ex_img:
        select = ex_img[class_poo]
        image_choice = st.radio("เลือกภาพตัวอย่าง", [f"Image {i+1}" for i in range(len(select))])
        img_index = int(image_choice.split()[1]) - 1
        img_path = select[img_index]
        st.image(img_path, caption=f"ภาพตัวอย่าง: {class_poo}", use_container_width=True)
        process_and_start_chat(img_path, key_suffix="test")

# --- 7. ระบบแชท (Chat Interface) ---
if "messages" in st.session_state and api_key_configured:
    st.subheader("💬 พูดคุยกับ AI ต่อเนื่อง")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["parts"][0])
    
    if prompt := st.chat_input("ถามคำถามเพิ่มเติมเกี่ยวกับสุขภาพได้ที่นี่..."):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("model"):
            with st.spinner("AI กำลังพิมพ์..."):
                try:
                    response = st.session_state.chat.send_message(prompt)
                    response_text = response.text
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "model", "parts": [response_text]})
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการสนทนา: {e}")

# --- ส่วนท้าย ---
st.subheader("", divider=True)
st.caption(":blue[Ai Builder Season 5 | Passawut Chutiparcharkij]")
