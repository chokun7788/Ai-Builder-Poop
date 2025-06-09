import streamlit as st
from fastai.vision.all import *
from PIL import Image 
import pandas as pd 
import plotly.express as px
from pathlib import Path
import sys
import pathlib
import google.generativeai as genai 

#ให้สามารถมใช้ได้กับ Windows และ Linux (ก่อน deploy จริง)
# _original_posix_path = None
# if sys.platform == "win32":
#     if hasattr(pathlib, 'PosixPath') and not isinstance(pathlib.PosixPath, pathlib.WindowsPath):
#         _original_posix_path = pathlib.PosixPath
#         pathlib.PosixPath = pathlib.WindowsPath


#llm gemini
api_key_configured = False
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
api_key_configured = True

# --- ฟังก์ชันสำหรับ "สร้างคำอธิบายเริ่มต้น" ---
def get_initial_explanation(stool_class):
    class_map = {
        "Blood": "มีเลือดปน (Blood)", "Diarrhea": "ท้องร่วง/ท้องเสีย (Diarrhea)",
        "Green": "สีเขียว (Green)", "Mucus": "มีมูกปน (Mucus)",
        "Normal": "ปกติ (Normal)", "Yellow": "สีเหลือง (Yellow)"
    }
    friendly_name = class_map.get(stool_class, stool_class)
    prompt = f"""
    ในฐานะผู้เชี่ยวชาญด้านสุขภาพเบื้องต้น โปรดให้ข้อมูลเกี่ยวกับอุจจาระประเภท "{friendly_name}" เพื่อเริ่มต้นการสนทนา
    กรุณาอธิบายโดยละเอียดเป็นภาษาไทย โดยแบ่งหัวข้อให้ชัดเจนดังนี้:
    1.  **สาเหตุที่เป็นไปได้:**
    2.  **ความเสี่ยงหรือโรคที่อาจเกี่ยวข้อง:**
    3.  **คำแนะนำเบื้องต้นและการดูแลตัวเอง:**
    **คำเตือนสำคัญ:** โปรดเน้นย้ำในตอนท้ายว่าข้อมูลนี้เป็นเพียงคำแนะนำเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้ และจบด้วยการบอกว่า "หากมีคำถามเพิ่มเติมเกี่ยวกับผลลัพธ์นี้ สามารถพิมพ์ถามได้เลยครับ"
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ขออภัย, เกิดข้อผิดพลาดในการเรียก AI เพื่อขอคำอธิบาย: {e}"


#model
MODEL_FILENAME = Path("convnextv2_thev1_best_for_good.pkl")
@st.cache_resource
def load_model(local_path):
    learn = load_learner(local_path)
    return learn
learn = load_model(MODEL_FILENAME)


#header
st.title("💩 :rainbow[Poop Classification & AI Chat]")
st.subheader("แยกประเภทอุจจาระ และพูดคุยถาม-ตอบกับ AI")
st.warning("⚠️ **ข้อควรระวัง:** ผลลัพธ์จาก AI นี้เป็นเพียงข้อมูลเบื้องต้นเพื่อการศึกษาเท่านั้น **ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้** หากมีอาการผิดปกติหรือกังวลใจ กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ")

#ส่วนสำหรับทำนายและตอบคำถาม(gemini)
def process_and_start_chat(image_source, key_suffix):
    if st.button("ทำนาย และ อธิบาย", key=key_suffix):
        with st.spinner('กำลังอธิบาย...'):
            pil_image = PILImage.create(image_source)
            pred_class, pred_idx, probs = learn.predict(pil_image)
            st.markdown(f"#### ผลลัพธ์: **{pred_class}**")
            st.markdown(f"##### ความน่าจะเป็น: **{probs[pred_idx]:.1%}**")
            df_probs = pd.DataFrame({'Class': learn.dls.vocab, 'Probability': probs.numpy() * 100})
            fig = px.pie(df_probs, values='Probability', names='Class', color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            
            
            with st.spinner('AI กำลังเตรียมคำอธิบายเริ่มต้น...'):
                initial_explanation = get_initial_explanation(pred_class)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                st.session_state.chat = model.start_chat(history=[])
                st.session_state.messages = [{"role": "model", "parts": [initial_explanation]}]

sec = st.selectbox("เลือกหมวดหมู่", ["อัปโหลดรูปเพื่อใช้งานจริง", "ทดลองใช้(สำหรับไม่มีรูป)"])

#first sec
if sec == "อัปโหลดรูปเพื่อใช้งานจริง":
    upload_file = st.file_uploader("อัปโหลดภาพของคุณ", type=["jpg", "jpeg", "png"])
    if upload_file:
        st.image(upload_file, caption="ภาพที่อัปโหลด", use_container_width=True)
        process_and_start_chat(upload_file, key_suffix="upload")
#sec sec
elif sec == "ทดลองใช้(สำหรับไม่มีรูป)":
    class_poo = st.selectbox("เลือกคลาสที่ต้องการทดสอบ", ["Blood", "Diarrhea", "Green", "Mucus", "Normal", "Yellow"])
    ex_img = {
        "Blood": [r"E:\MyFastAI_Project_E\Image\Blood\1.png",
                  r"E:\MyFastAI_Project_E\Image\Blood\2.jpg", 
                  r"E:\MyFastAI_Project_E\Image\Diarrhea\122879523_10222902185640115_5437019702589621427_n.jpg"],
        "Diarrhea": [r"E:\MyFastAI_Project_E\Image\Diarrhea\68621499_10158897636364968_929960603991146496_n.jpg",
                     r"E:\MyFastAI_Project_E\Image\Diarrhea\118258565_2797050553861531_5149781090231407705_n.jpg",
                     r"E:\MyFastAI_Project_E\Image\Diarrhea\362242137_646001919459_4084599573521026560_n.jpg"],
        "Green": [r"E:\MyFastAI_Project_E\Image\Green\470220113_1111404843690107_5400214401912539739_n.jpg",
                  r"E:\MyFastAI_Project_E\Image\Green\470210610_1614069995660748_7907742087399683339_n.jpg",
                  r"E:\MyFastAI_Project_E\Image\Green\363839965_10167704066005534_8500730712227200736_n.jpg"],
        "Mucus": [r"E:\MyFastAI_Project_E\Image\Mucus\does-this-look-like-it-could-be-worms-or-maybe-mucus-in-my-v0-6fvtr2ywdv4d1.png",
                  r"E:\MyFastAI_Project_E\Image\Mucus\mucus-in-stool-the-first-one-i-thought-was-a-parasite-but-v0-ocmq6pflaxib1.png",
                  r"E:\MyFastAI_Project_E\Image\Mucus\my-stormatch-intestine-make-noises-every-minute-what-should-v0-ortl442yi0ue1.png"],
        "Normal": [r"E:\MyFastAI_Project_E\Image\Normal\54.png",
                   r"E:\MyFastAI_Project_E\Image\Normal\52.png",
                   r"E:\MyFastAI_Project_E\Image\Normal\53.png"],
        "Yellow": [r"E:\MyFastAI_Project_E\Image\Yellow\470467721_122113376150620788_7483223442733841889_n.jpg", 
                   r"E:\MyFastAI_Project_E\Image\Yellow\480450326_1125657082688389_5418859568059391331_n.jpg", r"E:\MyFastAI_Project_E\Image\Yellow\481999682_1136825754904855_5806230666139878824_n.jpg"]
    }
    select = ex_img[class_poo]
    image_choice = st.radio("เลือกภาพที่ต้องการทำนาย", [f"Image {i+1}" for i in range(len(select))])
    img_index = int(image_choice.split()[1]) - 1
        
    img_path = select[img_index]
    st.image(img_path, caption=f"ภาพที่เลือก", use_container_width=True)
    process_and_start_chat(img_path, key_suffix="test")
    
#แชทเจนคำ
if "messages" in st.session_state and api_key_configured:
    st.subheader("พูดคุยกับ AI")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["parts"][0])
    
    if prompt := st.chat_input("ถามคำถามเพิ่มเติมเกี่ยวกับผลลัพธ์นี้..."):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("model"):
            with st.spinner("AI กำลังคิด..."):
                response = st.session_state.chat.send_message(prompt)
                response_text = response.text
                st.markdown(response_text)
        st.session_state.messages.append({"role": "model", "parts": [response_text]})

# --- ส่วนท้าย (คงเดิม) ---
st.subheader("", divider=True)
st.caption(":blue[Ai Builder Season 5]")
st.caption(":red[Passawut Chutiparcharkij | IG : passawut_727]")
