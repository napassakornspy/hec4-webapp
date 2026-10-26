import streamlit as st
import pandas as pd
import google.generativeai as genai
from hec4_model import HEC4

st.set_page_config(page_title="HEC-4 Web App", layout="wide")
st.title("🌊 โมเดลจำลองน้ำท่า HEC-4")

with st.sidebar:
    # เผื่อกรณีคีย์หาย จะได้ไม่ฟ้อง Error สีแดงเถือก
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    st.header("⚙️ พารามิเตอร์")
    n_years = st.number_input("จำนวนปีที่จำลอง", min_value=1, value=50)
    n_traces = st.number_input("จำนวนรูปแบบ (Traces)", min_value=1, value=5)
    missing_rate = st.slider("สัดส่วนจำลองข้อมูลแหว่ง", 0.0, 0.5, 0.1)

# ปุ่ม AI แนะนำการใช้งาน
if api_key:
    genai.configure(api_key=api_key)
    if st.button("💡 ให้ AI แนะนำวิธีใช้งานโปรแกรม"):
        with st.spinner("STAR กำลังพิมพ์คำแนะนำ..."):
            try:
                model_ai = genai.GenerativeModel("gemini-3.5-flash")
                prompt = "คุณคือ STAR ผู้ช่วยนักวิจัย ป.เอก ช่วยเขียนคำแนะนำวิธีใช้โปรแกรม HEC-4 Monthly Streamflow Simulation แบบสั้นๆ เข้าใจง่าย ไม่เกิน 3 ข้อ"
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: {e}")
else:
    st.warning("ใส่ API Key ด้านซ้ายมือก่อนนะคะ ถึงจะเรียกใช้เค้าได้")

# อัปเดตให้รองรับทั้ง CSV และ Excel 
uploaded_file = st.file_uploader("อัปโหลดไฟล์ข้อมูลน้ำท่า (CSV หรือ Excel)", type=["csv", "xlsx"])

# ใช้ Cache Resource เพื่อจำผลรัน เว็บจะได้ไม่อืดเวลากดปุ่มอื่น
@st.cache_resource(show_spinner=False)
def run_hec4_pipeline(file_buffer, years, traces, missing):
    model = HEC4(verbose=False)
    # ซอยคำสั่ง run_all ออกมาเป็นสเตป เพื่อให้ทำงานสอดคล้องกับ UI
    model.load_data(file_buffer, date_col="Date")
    model.generate_pseudo_missing(missing_rate=missing)
    model.infill_missing()
    model.compute_statistics()
    model.compute_correlation_matrices()
    model.compute_generation_matrices()
    model.generate_synthetic(n_years=years, n_traces=traces)
    model.validate_imputation()
    return model

if uploaded_file and st.button("🚀 รันโมเดล"):
    # ทำ Status Box ไล่สเตปตาม Flowchart
    with st.status("กำลังรันโมเดล HEC-4 ตาม Flowchart...", expanded=True) as status:
        st.write("📥 1/5 โหลดและจัดการรูปแบบข้อมูลดิบ...")
        st.write(f"🔍 2/5 สร้าง Pseudo-Missing Data (สัดส่วน {missing_rate})...")
        st.write("✨ 3/5 เติมเต็มข้อมูลแหว่ง (Regression & Mean Infill)...")
        st.write("📊 4/5 คำนวณค่าสถิติและเมทริกซ์ความสัมพันธ์เชิงพื้นที่...")
        st.write(f"🎲 5/5 สุ่มจำลองข้อมูลน้ำท่าสังเคราะห์ {n_years} ปี {n_traces} รูปแบบ...")
        
        # เรียกใช้งานฟังก์ชันที่หุ้ม Cache ไว้
        model = run_hec4_pipeline(uploaded_file, n_years, n_traces, missing_rate)
        
        status.update(label="ประมวลผลเสร็จสิ้น!", state="complete", expanded=False)
        
    st.success("จำลองข้อมูลเรียบร้อยแล้วตัว!")
    st.subheader("📈 ข้อมูลน้ำท่าสังเคราะห์ (Trace 001)")
    st.line_chart(model.synthetic_traces["trace_001"])
