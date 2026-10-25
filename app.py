import streamlit as st
import pandas as pd
import google.generativeai as genai
from hec4_model import HEC4

st.set_page_config(page_title="HEC-4 Web App", layout="wide")
st.title("🌊 โมเดลจำลองน้ำท่า HEC-4")

with st.sidebar:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.header("⚙️ พารามิเตอร์")
    n_years = st.number_input("จำนวนปีที่จำลอง", min_value=1, value=50)
    n_traces = st.number_input("จำนวนรูปแบบ (Traces)", min_value=1, value=5)
    missing_rate = st.slider("สัดส่วนจำลองข้อมูลแหว่ง", 0.0, 0.5, 0.1)

# ปุ่ม AI แนะนำการใช้งาน
if api_key:
    genai.configure(api_key=api_key)
    if st.button("💡 ให้ AI แนะนำวิธีใช้งานโปรแกรม"):
        with st.spinner("WARI กำลังพิมพ์คำแนะนำ..."):
            try:
                model_ai = genai.GenerativeModel("gemini-1.5-flash")
                prompt = "คุณคือ WARI ช่วยเขียนคำแนะนำวิธีใช้โปรแกรม HEC-4 Monthly Streamflow Simulation (มีช่องใส่จำนวนปี, Traces, missing rate และอัปโหลดไฟล์ CSV) ให้วิศวกรเข้าใจง่ายๆ สั้นๆ ไม่เกิน 3 ข้อ"
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
else:
    st.warning("ใส่ API Key ด้านซ้ายมือก่อนนะคะ ถึงจะเรียกใช้เค้าได้")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ข้อมูลน้ำท่าดิบ (คอลัมน์แรกชื่อ Date)", type=["csv"])

if uploaded_file and st.button("🚀 รันโมเดล"):
    with st.spinner("กำลังคำนวณเมทริกซ์และสุ่มข้อมูล..."):
        model = HEC4(verbose=False)
        model.load_data(uploaded_file, date_col="Date")
        model.run_all(n_years=n_years, n_traces=n_traces, missing_rate=missing_rate)
        
        st.success("ประมวลผลเสร็จสิ้น!")
        st.subheader("📈 ข้อมูลน้ำท่าสังเคราะห์ (Trace 001)")
        st.line_chart(model.synthetic_traces["trace_001"])
