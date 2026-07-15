# -*- coding: utf-8 -*-
"""
ส่วนที่ 3: Learning Curves เปรียบเทียบ 3 แบบจำลอง (จาก models.py)
  แกน x: จำนวนตัวอย่างฝึก N   แกน y: E_in (ชุดฝึก) และ E_out (จริง)
  ทดลอง 2 เงื่อนไข: ไม่มี noise และมี noise ~ N(0, sigma^2)

หมายเหตุ: linear + noise ที่ N=2 ค่า E_out จะระเบิดได้ (variance สูงเมื่อข้อมูลน้อย)
          เพราะบางรอบ x1 ใกล้ x2 มาก -> ความชันพุ่งสูง (ค่าจะหลุดกรอบ ylim)
"""

import numpy as np                                # ไลบรารีคำนวณเชิงตัวเลข
import matplotlib.pyplot as plt                   # ไลบรารีวาดกราฟ
from models import MODELS, TARGETS, predict       # ดึงแบบจำลอง เป้าหมาย และฟังก์ชันทำนายจากโมดูลกลาง

rng = np.random.default_rng(0)                    # ตัวสุ่ม seed=0 (ผลซ้ำได้)


def learning_curve(f, fit, Ns, sigma=0.0, trials=3000):
    """คืน (Ein เฉลี่ย, Eout เฉลี่ย) สำหรับแต่ละ N ใน Ns"""
    xs = np.linspace(-1, 1, 400)                  # กริดจุด x ไว้วัด E_out (ค่าคาดหวังนอกชุดฝึก)
    Ein, Eout = [], []                            # ลิสต์เก็บ error เฉลี่ยของแต่ละขนาดชุดฝึก N
    for N in Ns:                                  # วนทดลองทีละขนาดชุดฝึก
        ein = eout = 0.0                          # ตัวสะสม error ไว้เฉลี่ยตอนจบ
        for _ in range(trials):                   # ทำซ้ำหลายรอบเพื่อเฉลี่ยความสุ่ม
            x = rng.uniform(-1, 1, N)             # สุ่มชุดฝึก N จุด
            y = f(x) + sigma * rng.standard_normal(N)   # ค่าจริง + noise แบบเกาส์เซียน
            theta = fit(x, y)                     # fit แบบจำลองได้พารามิเตอร์ (a,b)
            ein  += np.mean((predict(theta, x) - y) ** 2)       # E_in: error บนชุดฝึก (เทียบ y ที่มี noise)
            eout += np.mean((predict(theta, xs) - f(xs)) ** 2)  # E_out: error เทียบฟังก์ชันจริงบนกริด
        Ein.append(ein / trials)                  # เฉลี่ย E_in ของขนาด N นี้
        Eout.append(eout / trials)                # เฉลี่ย E_out ของขนาด N นี้
    return np.array(Ein), np.array(Eout)


def main(fname="sin(pi*x)"):
    f = TARGETS[fname]                            # เลือกฟังก์ชันเป้าหมายตามชื่อ
    Ns = np.arange(2, 61, 2)                      # ขนาดชุดฝึกที่ทดลอง: 2, 4, 6, ..., 60
    sigmas = [0.0, 0.3]                           # สองเงื่อนไข: ไม่มี noise / มี noise sigma=0.3
    colors = {"constant": "tab:blue",            # กำหนดสีประจำแต่ละแบบจำลอง
              "linear": "tab:red",
              "lin-origin": "tab:green"}

    show_N = [2, 6, 20, 60]                       # ค่า N ที่จะยกมาโชว์ในตาราง
    idx = [list(Ns).index(n) for n in show_N]     # ตำแหน่งของ N เหล่านั้นใน Ns

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)  # 2 กราฟข้างกัน ใช้แกน y ร่วม
    for ax, sigma in zip(axes, sigmas):           # วนกราฟซ้าย(ไม่มี noise) และขวา(มี noise)
        # หัวตาราง E_out แยกต่อเงื่อนไข noise (คัดลอกลงสไลด์หน้า 9 ได้)
        print(f"\n=== Learning Curve (E_out)  f(x)={fname}, noise sigma={sigma} ===")
        print("model        | " + " ".join(f"N={n:<7}" for n in show_N))
        print("-" * (15 + 10 * len(show_N)))
        for name, fit in MODELS.items():          # วนแต่ละแบบจำลอง
            Ein, Eout = learning_curve(f, fit, Ns, sigma)   # คำนวณ learning curve
            ax.plot(Ns, Eout, "-",  color=colors[name], label=f"{name} Eout")  # เส้นทึบ = E_out
            ax.plot(Ns, Ein,  "--", color=colors[name], alpha=0.6,
                    label=f"{name} Ein")          # เส้นประ = E_in
            # พิมพ์ค่า E_out ที่ N ต่าง ๆ เป็นแถวตาราง
            print(f"{name:<12} | " + " ".join(f"{Eout[k]:9.3f}" for k in idx))
        if sigma > 0:                             # ถ้ามี noise ให้ลากเส้นขีดจำกัดล่าง (noise floor)
            ax.axhline(sigma**2, color="k", ls=":", lw=1,
                       label=f"noise floor σ²={sigma**2:.2f}")
        ax.set_title(f"f(x)={fname},  noise sigma={sigma}")  # ชื่อกราฟบอกเงื่อนไข
        ax.set_xlabel("N (training set size)")    # ป้ายแกน x
        ax.set_ylim(0, 1.2)                        # จำกัดแกน y (ค่าที่ระเบิดจะหลุดกรอบไปเอง)
        ax.legend(fontsize=8)                      # คำอธิบายเส้น
    axes[0].set_ylabel("Mean Squared Error")       # ป้ายแกน y (ใส่แค่กราฟซ้าย)

    plt.tight_layout()                            # จัดระยะไม่ให้ทับกัน
    plt.savefig("learning_curves.png", dpi=120)   # บันทึกรูป
    print("\nบันทึกรูป learning_curves.png แล้ว")


if __name__ == "__main__":                        # รันเฉพาะเมื่อสั่งไฟล์นี้ตรง ๆ
    main()
