# -*- coding: utf-8 -*-
"""
โมดูลกลาง: 3 แบบจำลอง + 2 ฟังก์ชันเป้าหมาย ใช้ร่วมกันทุกไฟล์
ทุก fit รับ (x, y) 1 มิติ แล้วคืน (a, b) เสมอ  ->  ทำนายด้วย a*x + b
(แบบจำลองไหนไม่มีพจน์นั้น ค่าจะเป็น 0)  fit ด้วย lstsq = normal equation
"""

import numpy as np                          # นำเข้า numpy สำหรับงานคำนวณเวกเตอร์/เมทริกซ์


def fit_const(x, y):                        # แบบจำลองค่าคงที่  h(x) = b  (เส้นแนวนอน)
    # สร้างเมทริกซ์ฟีเจอร์ที่มีแต่คอลัมน์ 1 (m,1) แล้วแก้สมการหา b ที่ทำให้ error น้อยสุด
    # ผลของ lstsq กับคอลัมน์ 1 ล้วน = ค่าเฉลี่ยของ y นั่นเอง
    b = np.linalg.lstsq(np.ones((len(x), 1)), y, rcond=None)[0]
    return np.array([0.0, b[0]])            # คืน (a=0, b) เพราะแบบนี้ไม่มีความชัน


def fit_linear(x, y):                       # แบบจำลองเชิงเส้น  h(x) = a*x + b
    # สร้างเมทริกซ์ฟีเจอร์ 2 คอลัมน์ [x, 1] เพื่อหา ทั้งความชัน a และจุดตัดแกน b
    X = np.column_stack([x, np.ones(len(x))])
    # lstsq แก้ least squares คืน [a, b] ที่ทำให้ผลรวมกำลังสองของ error น้อยสุด
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_linear_origin(x, y):                # แบบจำลองเชิงเส้นผ่านจุดกำเนิด  h(x) = a*x
    # ฟีเจอร์มีแค่คอลัมน์ x (บังคับจุดตัดแกน b=0) แล้วหาความชัน a อย่างเดียว
    a = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)[0]
    return np.array([a[0], 0.0])            # คืน (a, b=0)


# พจนานุกรมรวม 3 แบบจำลอง: ชื่อ -> ฟังก์ชัน fit  (ไว้วนลูปเรียกใช้ทีละตัว)
MODELS = {"constant":   fit_const,
          "linear":     fit_linear,
          "lin-origin": fit_linear_origin}

# พจนานุกรมฟังก์ชันเป้าหมายจริง 2 แบบ (สิ่งที่แบบจำลองพยายามเลียนแบบ)
TARGETS = {"sin(pi*x)": lambda x: np.sin(np.pi * x),   # คลื่นไซน์
           "x^2":       lambda x: x ** 2}              # พาราโบลา


def predict(theta, x):                      # ทำนายค่า y จากพารามิเตอร์ theta และอินพุต x
    """theta = (a, b)  ->  a*x + b"""
    return theta[0] * x + theta[1]          # ความชัน*x + จุดตัดแกน
