# -*- coding: utf-8 -*-
"""
โมดูลแบบจำลองของงานที่สาม — พหุนามดีกรี d (generalize จาก constant/linear ของงานก่อน)
แนวคิดเดียวกับ Sec_Work/models.py แต่โจทย์นี้ต้องการ "ทุกดีกรี d = 0..D" จึงย่อเหลือ
ฟังก์ชันเดียว fit_poly(x, y, d) ครอบคลุมทุกดีกรี (d=0 คือค่าคงที่, d=1 คือเชิงเส้น, ...)
"""

import warnings                                # ใช้ปิด warning ของ polyfit ที่ดีกรีสูง/จุดน้อย
import numpy as np                             # numpy สำหรับงานคำนวณเวกเตอร์/เมทริกซ์

# ดีกรีสูง + จุดน้อย ทำให้เมทริกซ์ near-singular -> polyfit เตือน RankWarning
# ปิดไว้เพราะ "อาการนั้นคือ overfitting ที่ตั้งใจให้เห็น" ไม่ใช่ข้อผิดพลาดของโค้ด
# (numpy 2.x ย้าย RankWarning ไป np.exceptions -> รองรับทั้งเก่า/ใหม่)
_RankWarning = getattr(np, "RankWarning", getattr(np.exceptions, "RankWarning", Warning))
warnings.simplefilter("ignore", _RankWarning)

# ฟังก์ชันเป้าหมายจริงของงานนี้: คลื่นไซน์ sin(pi*x) (โจทย์กำหนดมาแบบเดียว)
TARGET = lambda x: np.sin(np.pi * x)


def fit_poly(x, y, d):                         # ฟิตพหุนามดีกรี d บนข้อมูล (x, y)
    # np.polyfit = least squares หาสัมประสิทธิ์ที่ MSE ต่ำสุด, คืน [w_d, ..., w_1, w_0]
    # (สัมประสิทธิ์ดีกรีสูงสุดมาก่อน) -> d=0 จะได้ค่าคงที่ = ค่าเฉลี่ยของ y
    return np.polyfit(x, y, d)


def predict(coef, x):                          # ทำนายค่า y จากสัมประสิทธิ์พหุนาม coef
    return np.polyval(coef, x)                 # ประเมินพหุนาม รองรับทุกดีกรีในสูตรเดียว
