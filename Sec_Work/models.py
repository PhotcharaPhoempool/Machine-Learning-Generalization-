# -*- coding: utf-8 -*-
"""
โมดูลแบบจำลอง — *ดึง/คัดลอกมาจากงานแรก*
(First_Work_Bias_Variance_Learning_Curves/models.py)
คัดมาเฉพาะที่งานนี้ใช้: โมเดล constant, linear และฟังก์ชัน predict
(ตัด lin-origin ออก เพราะโจทย์งานที่สองใช้แค่ constant กับ linear)

ทุก fit รับ (x, y) 1 มิติ แล้วคืน theta = (a, b) เสมอ -> ทำนายด้วย a*x + b
"""

import numpy as np                          # numpy สำหรับงานคำนวณเวกเตอร์/เมทริกซ์


def fit_const(x, y):                        # แบบจำลองค่าคงที่  h(x) = b (เส้นแนวนอน)
    # lstsq กับเมทริกซ์คอลัมน์ 1 ล้วน = ค่าเฉลี่ยของ y (ค่าคงที่ที่ error น้อยสุด)
    b = np.linalg.lstsq(np.ones((len(x), 1)), y, rcond=None)[0]
    return np.array([0.0, b[0]])            # คืน (a=0, b) เพราะไม่มีความชัน


def fit_linear(x, y):                       # แบบจำลองเชิงเส้น  h(x) = a*x + b
    # ฟีเจอร์ 2 คอลัมน์ [x, 1] แล้ว lstsq หา [a, b] ที่ error กำลังสองน้อยสุด
    X = np.column_stack([x, np.ones(len(x))])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict(theta, x):                      # ทำนายค่า y จากพารามิเตอร์ theta และอินพุต x
    """theta = สัมประสิทธิ์พหุนาม (ดีกรีสูงสุดก่อน) -> ประเมินค่าด้วย np.polyval
    generalize จาก a*x+b เดิมมาเป็น polyval เพื่อให้ไฟล์ 'ศึกษาเพิ่มเติม'
    (poly3_extra_study.py) ใช้พหุนามดีกรีสูงร่วมได้ — ผลของ constant/linear เท่าเดิม
    เพราะ polyval([a,b],x)=a*x+b และ polyval([0,b],x)=b"""
    return np.polyval(theta, x)             # ประเมินพหุนามจากสัมประสิทธิ์ (รองรับทุกดีกรี)


# พจนานุกรมรวมแบบจำลอง: ชื่อ -> ฟังก์ชัน fit (ไว้วนลูปเรียกทีละตัว)
MODELS = {"constant": fit_const,
          "linear":   fit_linear}

# ฟังก์ชันเป้าหมายจริงของงานนี้: คลื่นไซน์ sin(pi*x) (โจทย์กำหนดมาแบบเดียว)
TARGET = lambda x: np.sin(np.pi * x)
