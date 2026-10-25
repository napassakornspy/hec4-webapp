import streamlit as st
import pandas as pd
from hec4_model import HEC4

st.set_page_config(page_title="HEC-4 Web App", layout="wide")
st.title("🌊 โมเดลจำลองน้ำท่า HEC-4")

# แถบตั้งค่าด้านข้าง
with st.sidebar:
    st.header("⚙️ พารามิเตอร์")
    n_years = st.number_input("จำนวนปีที่จำลอง", min_value=1, value=50)
    n_traces = st.number_input("จำนวนรูปแบบ (Traces)", min_value=1, value=5)
    missing_rate = st.slider("สัดส่วนจำลองข้อมูลแหว่ง", 0.0, 0.5, 0.1)

uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ข้อมูลน้ำท่าดิบ (คอลัมน์แรกต้องชื่อ Date)", type=["csv"])

if uploaded_file and st.button("🚀 รันโมเดล"):
    with st.spinner("กำลังคำนวณเมทริกซ์และสุ่มข้อมูล..."):
        # เรียกใช้งานคลาส HEC-4 จากไฟล์ hec4_model.py
        model = HEC4(verbose=False)
        model.load_data(uploaded_file, date_col="Date")
        model.run_all(n_years=n_years, n_traces=n_traces, missing_rate=missing_rate)
        
        st.success("ประมวลผลเสร็จสิ้น!")
        
        # ดึงข้อมูลมาพล็อตกราฟโชว์บนเว็บ
        st.subheader("📈 ข้อมูลน้ำท่าสังเคราะห์ (Trace 001)")
        trace_1 = model.synthetic_traces["trace_001"]
        st.line_chart(trace_1)
        
        # แสดงตารางข้อมูลที่อุดรอยรั่วแล้ว
        with st.expander("ดูตารางข้อมูลที่เติมเต็มแล้ว (Filled Data)"):
            st.dataframe(model.filled_data)