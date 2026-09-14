import pathlib
import pathlib
import sys
from pathlib import Path

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st
from fastai.vision.all import *


st.set_page_config(page_title="Poop Classification & AI Chat", page_icon="💩")


# -----------------------------------------------------------------------------
# Language settings
# -----------------------------------------------------------------------------
if "language" not in st.session_state:
    st.session_state.language = "th"
if "pdpa_consent" not in st.session_state:
    st.session_state.pdpa_consent = False


LANGUAGE = st.session_state.language

TEXT = {
    "th": {
        "page_title": "💩 :rainbow[Poop Classification & AI Chat]",
        "subtitle": "แยกประเภทอุจจาระ และพูดคุยถาม-ตอบกับ AI",
        "warning": "⚠️ **ข้อควรระวัง:** ผลลัพธ์จาก AI นี้เป็นเพียงข้อมูลเบื้องต้นเพื่อการศึกษาเท่านั้น **ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้** หากมีอาการผิดปกติหรือกังวลใจ กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ",
        "predict": "ทำนาย และ อธิบาย",
        "predicting": "กำลังอธิบาย...",
        "result": "ผลลัพธ์",
        "probability": "ความน่าจะเป็น",
        "preparing": "AI กำลังเตรียมคำอธิบายเริ่มต้น...",
        "category": "เลือกหมวดหมู่",
        "upload_mode": "อัปโหลดรูปเพื่อใช้งานจริง",
        "example_mode": "ทดลองใช้ (ตัวอย่างรูป)",
        "upload": "อัปโหลดภาพของคุณ",
        "uploaded_image": "ภาพที่อัปโหลด",
        "consent_title": "การยินยอมก่อนใช้ภาพ (PDPA)",
        "consent_alert": "⚠️ ก่อนถ่ายรูปหรืออัปโหลดภาพ กรุณาอ่านรายละเอียดและกดยินยอมก่อนใช้งาน",
        "consent_detail": "ภาพที่คุณถ่ายหรือเลือกอัปโหลดจะถูกนำมาใช้เพื่อวิเคราะห์ประเภทอุจจาระและสร้างคำอธิบายด้วย AI กรุณาอย่าอัปโหลดภาพที่มีข้อมูลส่วนบุคคลของผู้อื่นโดยไม่ได้รับอนุญาต",
        "consent_checkbox": "ฉันรับทราบและยินยอมให้ใช้ภาพเพื่อวัตถุประสงค์ดังกล่าว",
        "consent_accept": "ยินยอมและดำเนินการต่อ",
        "consent_required": "กรุณาติ๊กยินยอมก่อนดำเนินการต่อ",
        "input_method": "เลือกวิธีเพิ่มภาพ",
        "camera": "ถ่ายรูปด้วยกล้อง",
        "upload_from_device": "เลือกภาพจากเครื่อง",
        "camera_label": "กดเพื่อเปิดกล้องและถ่ายรูป",
        "camera_image": "ภาพที่ถ่ายจากกล้อง",
        "class": "เลือกคลาสที่ต้องการทดสอบ",
        "image": "เลือกภาพที่ต้องการทำนาย",
        "selected_image": "ภาพที่เลือก",
        "chat_title": "พูดคุยกับ AI",
        "chat_input": "ถามคำถามเพิ่มเติมเกี่ยวกับผลลัพธ์นี้...",
        "thinking": "AI กำลังคิด...",
        "switch": "English",
        "language_name": "ภาษาไทย",
        "class_names": {
            "Blood": "มีเลือดปน (Blood)",
            "Diarrhea": "ท้องร่วง/ท้องเสีย (Diarrhea)",
            "Green": "สีเขียว (Green)",
            "Mucus": "มีมูกปน (Mucus)",
            "Normal": "ปกติ (Normal)",
            "Yellow": "สีเหลือง (Yellow)",
        },
    },
    "en": {
        "page_title": "💩 :rainbow[Poop Classification & AI Chat]",
        "subtitle": "Classify stool and chat with AI",
        "warning": "⚠️ **Important:** This AI result is for preliminary educational information only and **cannot replace a diagnosis from a medical professional**. If you have unusual symptoms or concerns, please consult a healthcare professional.",
        "predict": "Predict and Explain",
        "predicting": "Preparing an explanation...",
        "result": "Result",
        "probability": "Probability",
        "preparing": "AI is preparing an initial explanation...",
        "category": "Choose a category",
        "upload_mode": "Upload an image",
        "example_mode": "Try an example image",
        "upload": "Upload your image",
        "uploaded_image": "Uploaded image",
        "consent_title": "Consent before using an image (PDPA)",
        "consent_alert": "⚠️ Before taking or uploading an image, please read the notice and give your consent.",
        "consent_detail": "The image you take or upload will be used to classify the stool and generate an AI explanation. Please do not upload an image containing another person’s personal data without permission.",
        "consent_checkbox": "I acknowledge and consent to the use of this image for the stated purpose.",
        "consent_accept": "Consent and continue",
        "consent_required": "Please check the consent box before continuing.",
        "input_method": "Choose how to add an image",
        "camera": "Take a photo with camera",
        "upload_from_device": "Choose an image from device",
        "camera_label": "Click to open the camera and take a photo",
        "camera_image": "Photo taken with camera",
        "class": "Choose a class to test",
        "image": "Choose an image to predict",
        "selected_image": "Selected image",
        "chat_title": "Chat with AI",
        "chat_input": "Ask a follow-up question about this result...",
        "thinking": "AI is thinking...",
        "switch": "ไทย",
        "language_name": "English",
        "class_names": {
            "Blood": "Blood",
            "Diarrhea": "Diarrhea",
            "Green": "Green",
            "Mucus": "Mucus",
            "Normal": "Normal",
            "Yellow": "Yellow",
        },
    },
}

T = TEXT[LANGUAGE]


def switch_language():
    st.session_state.language = "en" if st.session_state.language == "th" else "th"


if hasattr(st, "dialog"):
    @st.dialog(T["consent_title"])
    def show_pdpa_consent_dialog():
        st.warning(T["consent_alert"])
        st.write(T["consent_detail"])
        agreed = st.checkbox(T["consent_checkbox"], key="pdpa_checkbox")
        if st.button(T["consent_accept"], type="primary", use_container_width=True):
            if agreed:
                st.session_state.pdpa_consent = True
                st.rerun()
            else:
                st.error(T["consent_required"])
else:
    def show_pdpa_consent_dialog():
        st.warning(T["consent_alert"])
        st.info(T["consent_detail"])
        agreed = st.checkbox(T["consent_checkbox"], key="pdpa_checkbox")
        if agreed and st.button(T["consent_accept"], type="primary", key="pdpa_accept"):
            st.session_state.pdpa_consent = True
            st.rerun()


# ใช้กับ Windows/Linux (ก่อน deploy จริง)
_original_posix_path = None
if sys.platform == "win32":
    if hasattr(pathlib, "PosixPath") and not isinstance(pathlib.PosixPath, pathlib.WindowsPath):
        _original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath


# Gemini (Chat)
api_key_configured = False
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
api_key_configured = True


def get_initial_explanation(stool_class):
    friendly_name = T["class_names"].get(stool_class, stool_class)
    answer_language = "ภาษาไทย" if LANGUAGE == "th" else "English"
    prompt = f"""
    As a preliminary health information assistant, explain the stool type "{friendly_name}".
    Write the entire response in {answer_language}.
    Organize the response under these headings:
    1. Possible causes
    2. Potential risks or related conditions
    3. Basic advice and self-care
    Important warning: End by emphasizing that this is only preliminary information,
    cannot replace a diagnosis from a qualified doctor, and that the user can ask
    additional questions about this result.
    """

    model = genai.GenerativeModel("models/gemini-flash-lite-latest")
    response = model.generate_content(prompt)
    return response.text


MODEL_FILENAME = Path("convnextv2_thev1_best_for_good.pkl")


@st.cache_resource
def load_model(local_path):
    return load_learner(local_path)


learn = load_model(MODEL_FILENAME)


# -----------------------------------------------------------------------------
# Page header and language toggle
# -----------------------------------------------------------------------------
header_col, language_col = st.columns([5, 1])
with language_col:
    st.button(T["switch"], on_click=switch_language, use_container_width=True)

st.title(T["page_title"])
st.subheader(T["subtitle"])
st.warning(T["warning"])


# แสดงคำขอยินยอมเพียงครั้งเดียวต่อการเปิดเว็บ/session นี้
# เมื่อ Refresh หรือเปิดเว็บใหม่ Streamlit จะเริ่ม session ใหม่และแสดงอีกครั้ง
if not st.session_state.pdpa_consent:
    show_pdpa_consent_dialog()
    st.stop()


def process_and_start_chat(image_source, key_suffix):
    if st.button(T["predict"], key=key_suffix, use_container_width=True):
        with st.spinner(T["predicting"]):
            pil_image = PILImage.create(image_source)
            pred_class, pred_idx, probs = learn.predict(pil_image)
            st.markdown(f"#### {T['result']}: **{T['class_names'].get(pred_class, pred_class)}**")
            st.markdown(f"##### {T['probability']}: **{probs[pred_idx]:.1%}**")

            df_probs = pd.DataFrame(
                {"Class": learn.dls.vocab, "Probability": probs.numpy() * 100}
            )
            fig = px.pie(
                df_probs,
                values="Probability",
                names="Class",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

            with st.spinner(T["preparing"]):
                initial_explanation = get_initial_explanation(pred_class)
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                st.session_state.chat = model.start_chat(history=[])
                st.session_state.messages = [
                    {"role": "model", "parts": [initial_explanation]}
                ]


sec = st.selectbox(
    T["category"],
    [T["upload_mode"], T["example_mode"]],
)


if sec == T["upload_mode"]:
    input_method = st.radio(
        T["input_method"],
        [T["camera"], T["upload_from_device"]],
        horizontal=True,
    )
    image_source = None
    image_caption = T["uploaded_image"]

    if input_method == T["camera"]:
        image_source = st.camera_input(T["camera_label"])
        image_caption = T["camera_image"]
    else:
        image_source = st.file_uploader(T["upload"], type=["jpg", "jpeg", "png"])

    if image_source:
        st.image(image_source, caption=image_caption, use_container_width=True)
        process_and_start_chat(image_source, key_suffix="personal_image")

elif sec == T["example_mode"]:
    class_poo = st.selectbox(
        T["class"], ["Blood", "Diarrhea", "Green", "Mucus", "Normal", "Yellow"]
    )
    ex_img = {
        "Blood": [
            "Image/Blood/1.png",
            "Image/Blood/2.jpg",
            "Image/Blood/140398388_2159495844191715_8710468154881808551_n.jpg",
        ],
        "Diarrhea": [
            "Image/Diarrhea/68621499_10158897636364968_929960603991146496_n.jpg",
            "Image/Diarrhea/118258565_2797050553861531_5149781090231407705_n.jpg",
            "Image/Diarrhea/362242137_646001919459_4084599573521026560_n.jpg",
        ],
        "Green": [
            "Image/Green/470220113_1111404843690107_5400214401912539739_n.jpg",
            "Image/Green/470210610_1614069995660748_7907742087399683339_n.jpg",
            "Image/Green/363839965_10167704066005534_8500730712227200736_n.jpg",
        ],
        "Mucus": [
            "Image/Mucus/does-this-look-like-it-could-be-worms-or-maybe-mucus-in-my-v0-6fvtr2ywdv4d1.png",
            "Image/Mucus/mucus-in-stool-the-first-one-i-thought-was-a-parasite-but-v0-ocmq6pflaxib1.png",
            "Image/Mucus/my-stormatch-intestine-make-noises-every-minute-what-should-v0-ortl442yi0ue1.png",
        ],
        "Normal": ["Image/Normal/54.png", "Image/Normal/52.png", "Image/Normal/53.png"],
        "Yellow": [
            "Image/Yellow/470467721_122113376150620788_7483223442733841889_n.jpg",
            "Image/Yellow/480450326_1125657082688389_5418859568059391331_n.jpg",
            "Image/Yellow/481999682_1136825754904855_5806230666139878824_n.jpg",
        ],
    }

    select = ex_img[class_poo]
    image_choice = st.radio(T["image"], [f"Image {i + 1}" for i in range(len(select))])
    img_index = int(image_choice.split()[1]) - 1
    img_path = select[img_index]
    st.image(img_path, caption=T["selected_image"], use_container_width=True)
    process_and_start_chat(img_path, key_suffix="test")


# Chat
if "messages" in st.session_state and api_key_configured:
    st.subheader(T["chat_title"])
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["parts"][0])

    if prompt := st.chat_input(T["chat_input"]):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("model"):
            with st.spinner(T["thinking"]):
                response = st.session_state.chat.send_message(
                    f"Reply in {'Thai' if LANGUAGE == 'th' else 'English'}.\n\n{prompt}"
                )
                response_text = response.text
                st.markdown(response_text)
        st.session_state.messages.append({"role": "model", "parts": [response_text]})


st.subheader("", divider=True)
st.caption(":blue[Ai Builder Season 5]")
st.caption(":red[Passawut Chutiparcharkij | IG : passawut_727]")
