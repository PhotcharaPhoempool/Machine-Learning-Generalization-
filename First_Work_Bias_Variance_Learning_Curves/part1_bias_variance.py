# -*- coding: utf-8 -*-
"""
ส่วนที่ 1-2: หา Bias และ Variance ของ 3 แบบจำลอง (จาก models.py)
กับ 2 ฟังก์ชันเป้าหมาย  ข้อมูล: สุ่ม x จำนวน N=2 จุด จาก U[-1, 1]

  analytical -> ค่าคาดหวังตรง ๆ ด้วย numerical integration
  simulation -> Monte Carlo สุ่มหลายชุดแล้วเฉลี่ย

  bias     = E_x[ (gbar(x) - f(x))^2 ]
  variance = E_x[ E_D[ (g_D(x) - gbar(x))^2 ] ]
"""

import numpy as np                                # ไลบรารีคำนวณเชิงตัวเลข
import matplotlib.pyplot as plt                   # ไลบรารีวาดกราฟ
from scipy import integrate                       # ใช้ทำ numerical integration (อินทิเกรต)
from models import MODELS, TARGETS                # ดึงแบบจำลองและฟังก์ชันเป้าหมายจากโมดูลกลาง

rng = np.random.default_rng(42)                   # ตัวสุ่มแบบกำหนด seed=42 (ผลซ้ำได้ทุกครั้ง)


def simulate(f, fit, M=100_000, N=2):
    """Monte Carlo: fit ชุดข้อมูล M ชุด แล้วประมาณ bias, variance"""
    # วนสุ่มชุดข้อมูล M รอบ: แต่ละรอบสุ่ม x จำนวน N จุด แล้ว fit ได้พารามิเตอร์ (a,b)
    # เก็บทุกรอบเป็นเมทริกซ์ W ขนาด (M, 2)
    W = np.array([fit(x := rng.uniform(-1, 1, N), f(x)) for _ in range(M)])
    xs = np.linspace(-1, 1, 1001)                 # กริดจุด x 1001 จุด ไว้เฉลี่ยผลเหนือช่วง [-1,1]
    # คำนวณค่าทำนายของทุกแบบจำลองบนกริด: G[m] = a_m * xs + b_m  -> ได้เมทริกซ์ (M, 1001)
    G = np.outer(W[:, 0], xs) + W[:, 1:2]
    gbar = G.mean(axis=0)                         # gbar(x) = แบบจำลองเฉลี่ยเหนือทุกชุดข้อมูล
    bias = np.mean((gbar - f(xs)) ** 2)           # bias = เฉลี่ยกำลังสองของ (แบบจำลองเฉลี่ย - ค่าจริง)
    var  = np.mean(G.var(axis=0))                 # variance = ความแปรปรวนของแบบจำลอง เฉลี่ยเหนือ x
    return bias, var, xs, gbar, G                 # คืนค่าทั้งหมด (รวมข้อมูลไว้วาดกราฟ)


def analytical(f, fit):
    """สูตรค่าคาดหวัง: เฉลี่ย a, b เหนือ (x1,x2) ~ U[-1,1]^2 (ความหนาแน่น 1/4)"""
    # ฟังก์ชันช่วย: fit จากข้อมูล 2 จุด (x1, x2) แล้วคืนพารามิเตอร์ (a, b)
    ab = lambda x1, x2: fit(p := np.array([x1, x2]), f(p))
    # ฟังก์ชันช่วยหาค่าคาดหวัง E[g(a,b)] = อินทิเกรตสองชั้นเหนือ (x1,x2) คูณความหนาแน่นร่วม 1/4
    E = lambda g: integrate.dblquad(
        lambda x2, x1: g(*ab(x1, x2)) * 0.25, -1, 1, -1, 1)[0]

    Ea, Eb = E(lambda a, b: a), E(lambda a, b: b)                 # ค่าคาดหวังของ a และ b
    Ea2, Eb2, Eab = E(lambda a, b: a*a), E(lambda a, b: b*b), E(lambda a, b: a*b)  # โมเมนต์ที่ 2

    # bias = อินทิเกรต (แบบจำลองเฉลี่ย Ea*x+Eb ลบ ค่าจริง f(x)) ยกกำลังสอง คูณความหนาแน่น 1/2
    bias = integrate.quad(lambda x: (Ea*x + Eb - f(x))**2 * 0.5, -1, 1)[0]
    # variance = อินทิเกรต ( E[g(x)^2] - (E[g(x)])^2 ) คูณความหนาแน่น 1/2
    var  = integrate.quad(lambda x: (Ea2*x**2 + 2*Eab*x + Eb2
                                     - (Ea*x + Eb)**2) * 0.5, -1, 1)[0]
    return bias, var


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))    # เตรียมตาราง 2 แถว(เป้าหมาย) x 3 คอลัมน์(แบบจำลอง)

    for i, (fname, f) in enumerate(TARGETS.items()):   # วนแต่ละฟังก์ชันเป้าหมาย (แถว i)
        # พิมพ์หัวตารางแยกต่อฟังก์ชันเป้าหมาย (คัดลอกลงสไลด์ได้ทีละตาราง)
        print(f"\n=== ฟังก์ชันเป้าหมาย: {fname} ===")
        print(f"{'model':<12} | {'Bias²(ana)':>10} {'Bias²(sim)':>10} | "
              f"{'Var(ana)':>9} {'Var(sim)':>9} | {'Eout':>7}")
        print("-" * 68)
        for j, (mname, fit) in enumerate(MODELS.items()):  # วนแต่ละแบบจำลอง (คอลัมน์ j)
            b_ana, v_ana = analytical(f, fit)          # หา bias/var ด้วยวิธีอินทิเกรต
            b_sim, v_sim, xs, gbar, G = simulate(f, fit)   # หา bias/var ด้วยวิธีสุ่มจำลอง
            # พิมพ์ผลเทียบสองวิธี + Eout = bias + variance (error คาดหวังนอกชุดฝึก)
            print(f"{mname:<12} | {b_ana:10.4f} {b_sim:10.4f} | "
                  f"{v_ana:9.4f} {v_sim:9.4f} | {b_ana+v_ana:7.4f}")

            ax = axes[i, j]                            # เลือกช่องกราฟของคู่ (เป้าหมาย, แบบจำลอง) นี้
            for g in G[:60]:                           # วาดเส้นตัวอย่างแบบจำลอง 60 เส้น (สีเทาจาง)
                ax.plot(xs, g, color="gray", alpha=0.15, lw=0.8)
            sd = G.std(axis=0)                         # ส่วนเบี่ยงเบนมาตรฐานของแบบจำลองที่แต่ละ x
            # แรเงาแถบ gbar ± sd แสดงขอบเขตความแปรปรวน (variance)
            ax.fill_between(xs, gbar - sd, gbar + sd,
                            color="tab:blue", alpha=0.3, label="gbar ± sd")
            ax.plot(xs, f(xs), "g", lw=2, label="f(x)")        # เส้นเขียว = ฟังก์ชันจริง
            ax.plot(xs, gbar, "r--", lw=2, label="gbar(x)")    # เส้นแดงประ = แบบจำลองเฉลี่ย
            ax.set_ylim(-2, 2)                         # จำกัดแกน y ให้เห็นภาพชัด
            ax.set_title(f"f={fname}, {mname}\nbias={b_sim:.3f}, var={v_sim:.3f}")  # ชื่อกราฟ
            if i == 0 and j == 0:                      # ใส่คำอธิบายเส้น (legend) แค่ช่องแรก
                ax.legend(fontsize=8)

    plt.tight_layout()                                # จัดระยะช่องกราฟไม่ให้ทับกัน
    plt.savefig("bias_variance.png", dpi=120)         # บันทึกรูปลงไฟล์
    print("\nบันทึกรูป bias_variance.png แล้ว")


if __name__ == "__main__":                            # รันส่วนนี้เฉพาะเมื่อสั่งไฟล์นี้ตรง ๆ
    main()
