import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import google.generativeai as genai
from hec4_model import HEC4

st.set_page_config(page_title="HEC-4 Web App", layout="wide")
st.title("🌊 โมเดลจำลองข้อมูล HEC-4")

with st.expander("💡 คำแนะนำการเตรียมข้อมูลและตั้งค่าพารามิเตอร์ (คลิกเพื่ออ่าน)"):
    st.markdown("""
    **1. การเตรียมข้อมูล (CSV หรือ Excel)**
    * **คอลัมน์วันที่ (Date):** กำหนดคอลัมน์แรกให้เป็นวันที่ของข้อมูลรายเดือน (เช่น 2000-01-01) และตั้งชื่อหัวคอลัมน์ว่า "Date"
    * **คอลัมน์สถานี (Stations):** ถัดจากคอลัมน์วันที่ ให้จัด 1 คอลัมน์ต่อ 1 สถานีวัดน้ำ โดยใช้ชื่อสถานีเป็นหัวคอลัมน์
    * **รหัสข้อมูลที่ขาดหาย (Missing Codes):** สามารถใส่ค่าว่าง (NaN) หรือใช้ตัวเลขรหัส เช่น -999 หรือ -9999 โค้ดจะจัดการเติมข้อมูลอัตโนมัติให้

    **2. การตั้งค่าพารามิเตอร์**
    * **ประเภทข้อมูล:** เลือกว่าข้อมูลที่นำเข้าคือน้ำฝนหรือน้ำท่า เพื่อให้การวิเคราะห์และหน่วยถูกต้อง
    * **จำนวนปีที่จำลอง:** ความยาวของข้อมูลสังเคราะห์อนาคต
    * **จำนวนรูปแบบ (Traces):** จำนวนชุดข้อมูลจำลองที่เกิดจากการสุ่ม (Monte Carlo)
    * **สัดส่วนจำลองข้อมูลแหว่ง (Missing Rate):** เปอร์เซ็นต์ข้อมูลจริงที่จะจำลองว่าสูญหายเพื่อทดสอบความแม่นยำ
    """)

if "model_run_complete" not in st.session_state:
    st.session_state.model_run_complete = False
if "model_result" not in st.session_state:
    st.session_state.model_result = None

with st.sidebar:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    st.header("⚙️ พารามิเตอร์")
    data_type = st.selectbox("ประเภทข้อมูล:", ["น้ำท่า (Streamflow)", "น้ำฝน (Rainfall)"])
    n_years = st.number_input("จำนวนปีที่จำลอง", min_value=1, value=50)
    n_traces = st.number_input("จำนวนรูปแบบ (Traces)", min_value=1, value=5)
    missing_rate = st.slider("สัดส่วนจำลองข้อมูลแหว่ง", 0.0, 0.5, 0.1)
    
    if not api_key:
        st.warning("ใส่ API Key ใน Settings ก่อนนะ WARI ถึงจะช่วยวิเคราะห์ตอนท้ายได้")

uploaded_file = st.file_uploader(f"อัปโหลดไฟล์ข้อมูล{data_type} (CSV หรือ Excel)", type=["csv", "xlsx"])

#@st.cache_resource(show_spinner=False)
def run_hec4_pipeline(file_buffer, years, traces, missing, _status_box):
    model = HEC4(verbose=False)
    
    _status_box.info("📥 Stage 1/5: โหลดและจัดการรูปแบบข้อมูลดิบ...")
    model.load_data(file_buffer, date_col="Date")
    
    _status_box.info(f"🔍 Stage 2/5: สร้าง Pseudo-Missing Data (สัดส่วน {missing})...")
    model.generate_pseudo_missing(missing_rate=missing)
    
    _status_box.info("✨ Stage 3/5: เติมเต็มข้อมูลแหว่ง (Regression & Mean Infill)...")
    model.infill_missing()
    
    _status_box.info("📊 Stage 4/5: คำนวณค่าสถิติและเมทริกซ์ความสัมพันธ์เชิงพื้นที่...")
    model.compute_statistics()
    model.compute_correlation_matrices()
    model.compute_generation_matrices()
    
    _status_box.info(f"🎲 Stage 5/5: สุ่มจำลองข้อมูลสังเคราะห์ {years} ปี {traces} รูปแบบ...")
    model.generate_synthetic(n_years=years, n_traces=traces)
    model.validate_imputation()
    
    return model

if uploaded_file and st.button("🚀 รันโมเดล"):
    status_box = st.empty()
    st.session_state.model_result = run_hec4_pipeline(uploaded_file, n_years, n_traces, missing_rate, status_box)
    st.session_state.model_run_complete = True
    status_box.success("✅ ประมวลผลเสร็จสิ้นเรียบร้อยแล้ว!")

if st.session_state.model_run_complete:
    model = st.session_state.model_result
    
    st.subheader(f"📈 ข้อมูล{data_type}สังเคราะห์ (Trace 001)")
    st.line_chart(model.synthetic_traces["trace_001"])
    
    csv_data = model.synthetic_traces["trace_001"].to_csv().encode('utf-8')
    st.download_button(
        label="📥 ดาวน์โหลดข้อมูล (CSV)",
        data=csv_data,
        file_name="synthetic_trace_001.csv",
        mime="text/csv"
    )
    
    st.subheader("📊 สถิติเชิงพรรณนา")
    st.dataframe(model.synthetic_traces["trace_001"].describe())

    st.subheader("⚖️ Double Mass Analysis")
    st.write("ตรวจสอบความสม่ำเสมอของข้อมูล (Consistency) เทียบกับค่าเฉลี่ยสถานีรอบข้าง")
    
    target_station = st.selectbox("เลือกสถานีที่ต้องการตรวจสอบ:", model.stations)
    if target_station:
        base_stations = [s for s in model.stations if s != target_station]
        target_cumsum = model.filled_data[target_station].cumsum()
        base_avg = model.filled_data[base_stations].mean(axis=1)
        base_cumsum = base_avg.cumsum()
        
        plot_type = "Streamflow" if "น้ำท่า" in data_type else "Rainfall"
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(base_cumsum, target_cumsum, marker='o', markersize=4, linestyle='-', color='#1f77b4', label='Double Mass Curve')
        ax.set_title(f"{target_station} vs Average of {len(base_stations)} Base Stations", fontsize=12)
        ax.set_xlabel("Cumulative Mean of Base Stations")
        ax.set_ylabel(f"Cumulative {plot_type} at {target_station}")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
     
    st.divider()
    st.subheader("🏆 คัดเลือก Trace ที่เหมาะสมที่สุด (Best Representative Trace)")
    
    historical_mean = model.filled_data.mean().mean() 
    best_trace_name = None
    min_diff = float('inf')
    
    for trace_name, df_trace in model.synthetic_traces.items():
        trace_mean = df_trace.mean().mean()
        diff = abs(trace_mean - historical_mean)
        if diff < min_diff:
            min_diff = diff
            best_trace_name = trace_name
            
    st.success(f"✨ โปรแกรมคัดเลือก **{best_trace_name}** มาให้แล้วจ้ะ!")
    st.write(f"เหตุผลเบื้องต้น: มีค่าเฉลี่ยปริมาณ{data_type}ใกล้เคียงกับข้อมูลในอดีตมากที่สุด")
    
    # 1. ย้ายปุ่มดาวน์โหลดออกมาวางข้างนอก เพื่อไม่ให้ Streamlit รวน
    csv_best = model.synthetic_traces[best_trace_name].to_csv().encode('utf-8')
    st.download_button(
        label=f"📥 ดาวน์โหลดเฉพาะ {best_trace_name} ไปใช้งาน (CSV)",
        data=csv_best,
        file_name=f"best_{best_trace_name}.csv",
        mime="text/csv"
    )
    
    # 2. ปุ่มให้ WARI วิเคราะห์ผลแยกไว้ต่างหาก
    if api_key:
        if st.button(f"🧠 ให้ WARI สรุปเหตุผลเจาะลึกของ {best_trace_name}"):
            with st.spinner("WARI กำลังเรียบเรียงเหตุผลให้นะคะ..."):
                try:
                    best_trace_stats = model.synthetic_traces[best_trace_name].describe().to_string()
                    # (ใส่โค้ด Prompt พิมพ์ prompt = f"""...""" และโค้ดเรียก AI เหมือนเดิมเลยจ้ะ)
                    
                    genai.configure(api_key=api_key)
                    model_ai = genai.GenerativeModel("gemini-3.5-flash")
                    response = model_ai.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"การเชื่อมต่อ API มีปัญหา: {e}")
