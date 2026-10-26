import streamlit as st
import pandas as pd
import google.generativeai as genai
from hec4_model import HEC4

st.set_page_config(page_title="HEC-4 Web App", layout="wide")
st.title("🌊 โมเดลจำลองน้ำท่า HEC-4")

# ใช้ st.expander ซ่อนคำแนะนำไว้ หน้าจอจะได้ดูสะอาดตาจ้ะ
with st.expander("💡 คำแนะนำการเตรียมข้อมูลและตั้งค่าพารามิเตอร์ (คลิกเพื่ออ่าน)"):
    st.markdown("""
    **1. การเตรียมข้อมูล (CSV หรือ Excel)**
    ข้อมูลที่ต้องเตรียมสำหรับนำเข้าโปรแกรม HEC-4 ต้องจัดโครงสร้างให้เป็นข้อมูลอนุกรมเวลา (Time Series) ดังนี้:
    * **คอลัมน์วันที่ (Date):** กำหนดคอลัมน์แรกให้เป็นวันที่ของข้อมูลรายเดือน (เช่น 2000-01-01) และตั้งชื่อหัวคอลัมน์ว่า "Date"
    * **คอลัมน์สถานี (Stations):** ถัดจากคอลัมน์วันที่ ให้จัด 1 คอลัมน์ต่อ 1 สถานีวัดน้ำ โดยใช้ชื่อสถานีเป็นหัวคอลัมน์ (เช่น P1, P7A) แล้วใส่ตัวเลขปริมาณน้ำท่าไล่ลงมา
    * **รหัสข้อมูลที่ขาดหาย (Missing Codes):** เดือนไหนที่ข้อมูลแหว่ง ปายสามารถใส่ค่าว่าง (NaN) หรือใช้ตัวเลขรหัส เช่น -999 หรือ -9999 ได้เลย โค้ดจะจัดการเติมข้อมูลอัตโนมัติให้ค่ะ

    **2. การตั้งค่าพารามิเตอร์ (ด้านซ้ายมือ)**
    * **จำนวนปีที่จำลอง (Number of Simulation Years):** ความยาวของข้อมูลน้ำท่าสังเคราะห์อนาคต ควรปรับตาม "อายุการออกแบบ (Design Life)" ของโปรเจกต์ เช่น 30, 50 หรือ 100 ปี
    * **จำนวนรูปแบบ (Traces):** จำนวนชุดข้อมูลจำลองที่เกิดจากการสุ่ม (Monte Carlo) การสร้างหลาย Traces ช่วยครอบคลุมความเสี่ยง ปกติงานวิทยานิพนธ์ควรใช้ 10-50 Traces ยิ่งเยอะสถิติยิ่งแน่น แต่คอมจะใช้เวลาคำนวณนานขึ้นนิดนึงค่ะ
    * **สัดส่วนจำลองข้อมูลแหว่ง (Missing Rate):** เปอร์เซ็นต์ของข้อมูลจริงที่จะ "แกล้งซ่อนไว้" (Pseudo-Missing) เพื่อทดสอบหาค่าความแม่นยำ ควรตั้งไว้ที่ 0.10 ถึง 0.20 (10-20%) ถ้าตั้งเป็น 0 โมเดลจะข้ามการทดสอบความแม่นยำไปเลย
    """)

# จัดการ State ป้องกันเว็บรีเฟรชแล้วกราฟกับปุ่ม AI หาย
if "model_run_complete" not in st.session_state:
    st.session_state.model_run_complete = False
if "model_result" not in st.session_state:
    st.session_state.model_result = None

with st.sidebar:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    st.header("⚙️ พารามิเตอร์")
    n_years = st.number_input("จำนวนปีที่จำลอง", min_value=1, value=50)
    n_traces = st.number_input("จำนวนรูปแบบ (Traces)", min_value=1, value=5)
    missing_rate = st.slider("สัดส่วนจำลองข้อมูลแหว่ง", 0.0, 0.5, 0.1)
    
    if not api_key:
        st.warning("ใส่ API Key ใน Settings ก่อนนะคะ เค้าถึงจะช่วยวิเคราะห์ตอนท้ายได้")

uploaded_file = st.file_uploader("อัปโหลดไฟล์ข้อมูลน้ำท่า (CSV หรือ Excel)", type=["csv", "xlsx"])

@st.cache_resource(show_spinner=False)
def run_hec4_pipeline(file_buffer, years, traces, missing):
    model = HEC4(verbose=False)
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
    with st.status("กำลังรันโมเดล HEC-4 ตาม Flowchart...", expanded=True) as status:
        st.write("📥 1/5 โหลดและจัดการรูปแบบข้อมูลดิบ...")
        st.write(f"🔍 2/5 สร้าง Pseudo-Missing Data (สัดส่วน {missing_rate})...")
        st.write("✨ 3/5 เติมเต็มข้อมูลแหว่ง (Regression & Mean Infill)...")
        st.write("📊 4/5 คำนวณค่าสถิติและเมทริกซ์ความสัมพันธ์เชิงพื้นที่...")
        st.write(f"🎲 5/5 สุ่มจำลองข้อมูลน้ำท่าสังเคราะห์ {n_years} ปี {n_traces} รูปแบบ...")
        
        # รันโมเดลและเก็บลง Session State
        st.session_state.model_result = run_hec4_pipeline(uploaded_file, n_years, n_traces, missing_rate)
        st.session_state.model_run_complete = True
        
        status.update(label="ประมวลผลเสร็จสิ้น!", state="complete", expanded=False)

# แสดงผลและเรียกใช้ WARI เมื่อรันโมเดลเสร็จแล้วเท่านั้น
if st.session_state.model_run_complete:
    model = st.session_state.model_result
    st.success("จำลองข้อมูลเรียบร้อยแล้วค่า!")
    st.subheader("📈 ข้อมูลน้ำท่าสังเคราะห์ (Trace 001)")
    st.line_chart(model.synthetic_traces["trace_001"])

    # ย้าย AI มาโชว์วิเคราะห์ผลตอนจบ
    if api_key:
        st.subheader("🧠 ให้ WARI ช่วยวิเคราะห์ผลลัพธ์")
        if st.button("คลิกเพื่อให้ WARI สรุปข้อมูล"):
            with st.spinner("เค้ากำลังคำนวณและสรุปผลให้อยู่นะ..."):
                try:
                    genai.configure(api_key=api_key)
                    model_ai = genai.GenerativeModel("gemini-3.5-flash")
                    # ดึงสถิติพื้นฐานส่งให้ AI วิเคราะห์
                    trace1_stats = model.synthetic_traces["trace_001"].describe().to_string()
                    prompt = f"คุณคือ WARI เพศหญิง ผู้ช่วยวิศวกรแหล่งน้ำ/วิศวกรชลประทาน/นักอุทกวิทยา ช่วยวิเคราะห์ผลสถิติของข้อมูลน้ำท่าสังเคราะห์ Trace 1 จากโมเดล HEC-4 ชุดนี้ให้หน่อย สรุปสั้นๆ ให้เห็นแนวโน้ม ค่าสูงสุด/ต่ำสุด หรือจุดที่วิศวกรอุทกวิทยาควรระวังในการออกแบบโครงสร้าง:\n\n{trace1_stats}"
                    
                    response = model_ai.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"API ดื้อจ้ะ เชื่อมต่อไม่ได้: {e}")
