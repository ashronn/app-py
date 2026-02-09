import streamlit as st
import pandas as pd
import os
import re

# --- การตั้งค่าหน้ากระดาษ ---
st.set_page_config(page_title="Sensor Quality Analysis", layout="wide")

# --- ส่วนของ Sidebar ---
with st.sidebar:
    st.title("📂 การจัดการไฟล์")
    uploaded_files = st.file_uploader("เลือกไฟล์ CSV", type=['csv'], accept_multiple_files=True)
    st.divider()
    st.write("⚙️ **Settings**")
    # ปรับเกณฑ์ความต่างจากค่าปกติ
    spike_limit = st.number_input("เกณฑ์การพุ่งผิดปกติ (Value Spike)", value=100.0)
    st.info(f"ระบบจะนับเป็น Error เมื่อ:\n1. ค่าพุ่งเกินกว่าค่าปกติรอบข้าง (Median) เกิน {spike_limit}\n2. ค่าสูงเกิน 6600")

# --- ฟังก์ชันวิเคราะห์ข้อมูล ---
def get_analysis_data(df, suffix, dht_cols, piera_cols, limit):
    # เตรียมข้อมูลเบื้องต้น
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y-%H-%M-%S', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime') 
    
    # --- 1. หา Time Gap (ช่วงที่ข้อมูลหายนานที่สุด) ---
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()
    max_gap = df['time_diff'].max() - 1 if df['time_diff'].max() > 1 else 0
    gap_info = "ไม่มีข้อมูลขาดหาย"
    if max_gap > 0:
        gap_idx = df['time_diff'].idxmax()
        gap_end_time = df.loc[gap_idx, 'datetime']
        prev_idx_pos = df.index.get_loc(gap_idx) - 1
        gap_start_time = df.iloc[prev_idx_pos]['datetime']
        gap_info = f"{int(max_gap)} วินาที ({gap_start_time.strftime('%H:%M:%S')} - {gap_end_time.strftime('%H:%M:%S')})"

    # --- 2. ตรวจสอบ Outlier (ค่าพุ่งผิดปกติ) ---
    pm_col = next((c for c in df.columns if 'PM2' in c and '5' in c), None)
    error_info = "ไม่พบค่าผิดปกติ"
    total_errors = 0
    
    if pm_col:
        # --- Logic ใหม่: เทียบค่าปัจจุบันกับ Rolling Median ---
        pm_valid = df[['datetime', pm_col]].dropna().copy()
        
        # หาค่าปกติรอบข้าง (ใช้ 11 แถวรอบๆ)
        pm_valid['baseline'] = pm_valid[pm_col].rolling(window=11, center=True, min_periods=1).median()
        
        # เช็คว่าค่าปัจจุบัน 'พุ่งสูง' กว่าค่าปกติรอบข้างหรือไม่
        # (ใช้ค่าลบกันตรงๆ เพื่อดูว่ามันโดดขึ้นไป ไม่ใช่ตอนมันตกลงมา)
        pm_valid['is_error'] = (pm_valid[pm_col] > 6600) | ((pm_valid[pm_col] - pm_valid['baseline']) > limit)
        
        # นับจำนวน Error
        total_errors = pm_valid['is_error'].sum()
        if total_errors > 0:
            err_times = pm_valid[pm_valid['is_error'] == True]['datetime'].dt.strftime('%H:%M:%S').unique()
            error_info = f"พบ {total_errors} ครั้ง (ตัวอย่าง: {', '.join(err_times[:3])})"
        
        # นำสถานะ Error กลับไปรวมกับตารางหลัก
        df = df.merge(pm_valid[['datetime', 'is_error']], on='datetime', how='left')
        df[f'pm_error_{suffix}'] = df['is_error'].fillna(False).astype(int)
    else:
        df[f'pm_error_{suffix}'] = 0
        total_errors = 0

    # --- 3. เตรียมข้อมูลสถานะสำหรับแสดงผลรายนาที ---
    status_df = pd.DataFrame()
    status_df['datetime'] = df['datetime']
    status_df[f'has_dt_{suffix}'] = 1
    status_df[f'has_dht_{suffix}'] = df[dht_cols].notnull().any(axis=1).astype(int)
    status_df[f'has_piera_{suffix}'] = df[piera_cols].notnull().any(axis=1).astype(int)
    status_df[f'both_up_{suffix}'] = ((status_df[f'has_dht_{suffix}'] == 1) & (status_df[f'has_piera_{suffix}'] == 1)).astype(int)
    status_df[f'pm_error_{suffix}'] = df[f'pm_error_{suffix}']
    
    return status_df.groupby('datetime').max(), gap_info, total_errors, error_info

# --- ส่วนเนื้อหาหลัก ---
st.title("📊 Sensor Data Quality & Outlier Analysis")

if uploaded_files:
    data_groups = {} 
    for file in uploaded_files:
        name = file.name
        date_match = re.search(r'(\d{2}-\d{2}-\d{4})', name)
        if date_match:
            date_key = date_match.group(1)
            point_id = "P1" if "Point01" in name else "P2"
            if date_key not in data_groups: data_groups[date_key] = {}
            data_groups[date_key][point_id] = file

    for date_key in sorted(data_groups.keys()):
        st.subheader(f"📅 วันที่: {date_key}")
        group = data_groups[date_key]
        
        if 'P1' in group and 'P2' in group:
            df1, df2 = pd.read_csv(group['P1']), pd.read_csv(group['P2'])
            dht_cols, piera_cols = ['humidity', 'temperature'], [c for c in df1.columns if c.startswith('PC') or c.startswith('PM')]

            status_p1, gap_p1, err_count_p1, err_time_p1 = get_analysis_data(df1, 'P1', dht_cols, piera_cols, spike_limit)
            status_p2, gap_p2, err_count_p2, err_time_p2 = get_analysis_data(df2, 'P2', dht_cols, piera_cols, spike_limit)
            
            combined = pd.concat([status_p1, status_p2], axis=1)
            analysis_1min = combined.fillna(0).resample('1min').sum()
            analysis_1min['missing_P1'] = (60 - analysis_1min['has_dt_P1']).clip(lower=0)
            analysis_1min['missing_P2'] = (60 - analysis_1min['has_dt_P2']).clip(lower=0)

            st.info(f"📈 **Raw Data:** P1: {len(df1):,} แถว | P2: {len(df2):,} แถว")
            
            # --- แถวที่ 1 ---
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("ข้อมูลเข้าเฉลี่ย/นาที", f"{analysis_1min['has_dt_P1'].mean():.2f} | {analysis_1min['has_dt_P2'].mean():.2f}")
            with col2: st.metric("Avg DHT", f"{analysis_1min['has_dht_P1'].mean():.2f} | {analysis_1min['has_dht_P2'].mean():.2f}")
            with col3: st.metric("Avg Piera", f"{analysis_1min['has_piera_P1'].mean():.2f} | {analysis_1min['has_piera_P2'].mean():.2f}")
            with col4: st.metric("ครบ 2 เซนเซอร์", f"{analysis_1min['both_up_P1'].mean():.2f} | {analysis_1min['both_up_P2'].mean():.2f}")

            # --- แถวที่ 2 ---
            st.markdown("#### ⚠️ ตรวจสอบคุณภาพข้อมูล (Quality Check)")
            c_miss, c_gap, c_err = st.columns([1, 1.5, 1])
            with c_miss:
                st.metric("ข้อมูลสูญหายรวม (P1|P2)", f"{int(analysis_1min['missing_P1'].sum()):,} | {int(analysis_1min['missing_P2'].sum()):,}", delta="วินาที", delta_color="inverse")
            with c_gap:
                st.write(f"⌛ **ช่วงที่ข้อมูลหายนานที่สุด:**")
                st.write(f"• P1: {gap_p1}")
                st.write(f"• P2: {gap_p2}")
            with c_err:
                st.metric("Total Errors (Outliers)", f"{int(err_count_p1):,} | {int(err_count_p2):,}", delta="จุดที่พุ่ง", delta_color="inverse")
                st.write(f"🕒 **เวลาที่พบค่าผิดปกติ:**")
                st.caption(f"P1: {err_time_p1}")
                st.caption(f"P2: {err_time_p2}")

            with st.expander("🔍 ดูรายละเอียดรายนาทีและดาวน์โหลด"):
                st.dataframe(analysis_1min[['has_dt_P1', 'missing_P1', 'pm_error_P1', 'has_dt_P2', 'missing_P2', 'pm_error_P2']], use_container_width=True)
                csv = analysis_1min.to_csv().encode('utf-8')
                st.download_button("📥 ดาวน์โหลดสรุป (.csv)", data=csv, file_name=f"Summary_{date_key}.csv", key=f"dl_{date_key}")
            st.divider()
        else:
            st.warning(f"⚠️ วันที่ {date_key} ข้อมูลไม่ครบ")
else:
    st.info("💡 กรุณาอัปโหลดไฟล์ที่ Sidebar")