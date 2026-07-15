# -*- coding: utf-8 -*-
"""
=== การศึกษาเพิ่มเติม (นอกโจทย์) : เพิ่มโมเดล polynomial ดีกรี 3 ===

โจทย์งานที่สองกำหนดแค่ 2 โมเดล (constant, linear) ซึ่งทำครบแล้วใน perf_estimators.py
ไฟล์นี้ *แยกออกมาต่างหาก* เพื่อศึกษาเพิ่มว่า ถ้าโมเดลยืดหยุ่นขึ้น (poly3)
ตัวประมาณ E_out ทั้ง 3 แบบจะมีพฤติกรรมอย่างไร
  คาด: poly3 overfit ได้ -> resub มองโลกแง่ดี "ยิ่งกว่าเดิม" และ variance สูงกว่า linear

** เน้นใช้ซ้ำของเดิม (reuse) **
  - โมเดล constant/linear ดึงจาก models.py (ซึ่งดึงมาจากงานแรกอีกที)
  - ฟังก์ชันตัวประมาณ/การจำลองทั้งหมด ดึงจากไฟล์งานหลัก perf_estimators.py โดยตรง
    (ใช้ได้เลยเพราะ predict ใน models.py ตอนนี้เป็น np.polyval รองรับพหุนามทุกดีกรี)
  - การ import perf_estimators ไม่ทำให้ main() ของมันรัน (มี __main__ guard อยู่)
"""

import warnings                                    # ใช้ปิด warning ของ polyfit ดีกรีสูง
import numpy as np                                 # ไลบรารีคำนวณเชิงตัวเลข
import matplotlib.pyplot as plt                    # ไลบรารีวาดกราฟ

# --- reuse: โมเดลเดิมจาก models.py ---
from models import fit_const, fit_linear           # โมเดล constant, linear (มาจากงานแรก)
# --- reuse: ตัวประมาณ + ตัวจำลอง + ค่าคงที่ จากไฟล์งานหลัก perf_estimators.py ---
from perf_estimators import (est_resub, est_holdout, est_kfold, run_trials,
                             stats_vs_true, make_data, true_eout,
                             N, SIGMA, M_DATASETS, HOLDOUT_RATIO, K_FOLD)

# numpy 2.x ย้าย RankWarning ไป numpy.exceptions -> รองรับทั้งเก่า/ใหม่
try:
    from numpy.exceptions import RankWarning       # numpy >= 2.0
except ImportError:
    RankWarning = np.RankWarning                   # numpy < 2.0
warnings.simplefilter("ignore", RankWarning)       # ปิด warning ตอน polyfit จุดน้อย (ผลยังใช้ได้)


# ---------- ของใหม่เฉพาะไฟล์นี้ ----------
# โมเดลพหุนามดีกรี 3: np.polyfit หา สปส. 4 ตัว (ดีกรีสูงสุดก่อน) แบบ least-squares
# หมายเหตุ: poly3 ต้องมีจุดฝึก >= 4 จุด — ค่าที่กวาดในไฟล์นี้ (n>=10, train subset >= 5) ปลอดภัย
fit_poly3 = lambda x, y: np.polyfit(x, y, 3)

# ชุดโมเดลขยาย: เพิ่ม poly3 ต่อท้าย constant/linear
MODELS_EXT = {"constant": fit_const,
              "linear":   fit_linear,
              "poly3":    fit_poly3}

METHODS = ["resub", "holdout", "kfold"]            # ชื่อ 3 วิธี ไว้วนพิมพ์/วาด
COLORS = ["tab:red", "tab:green", "tab:blue"]      # สีประจำแต่ละวิธี


# ======================= (a) ประมาณบนข้อมูล 1 ชุด =======================

def part_a(rng):
    """ประมาณ E_out ด้วย 3 วิธีบนข้อมูล 1 ชุด ต่อทั้ง 3 โมเดล (รวม poly3)"""
    print("\n=== (a) ประมาณ E_out บนข้อมูล 1 ชุด — รวม poly3 (n=%d, sigma=%.2f) ===" % (N, SIGMA))
    print(f"{'model':<10} | {'resub':>8} {'holdout':>8} {'kfold':>8} | {'E_out(จริง)':>10}")
    print("-" * 54)
    for name, fit in MODELS_EXT.items():           # วนทุกโมเดลในชุดขยาย
        x, y = make_data(N, SIGMA, rng)            # สุ่มข้อมูล 1 ชุด
        r = est_resub(x, y, fit)                   # ค่าประมาณ resubstitution
        h = est_holdout(x, y, fit, HOLDOUT_RATIO, rng)   # ค่าประมาณ holdout
        cv = est_kfold(x, y, fit, K_FOLD, rng)     # ค่าประมาณ k-fold
        t = true_eout(fit(x, y), SIGMA)            # E_out จริงของโมเดลเต็มชุด
        print(f"{name:<10} | {r:8.4f} {h:8.4f} {cv:8.4f} | {t:10.4f}")
    print("สังเกต: poly3 ยืดหยุ่นสูง -> resub ยิ่งต่ำกว่า E_out จริง (overfit)")


# ======================= (b) bias/variance/MSE + histogram =======================

def part_b(rng):
    """ทำซ้ำ M ชุด -> bias/variance/MSE ของแต่ละวิธี ต่อทั้ง 3 โมเดล + วาด histogram"""
    print("\n=== (b) bias/variance/MSE — รวม poly3 (ทำซ้ำ %d ชุด, n=%d, sigma=%.2f) ==="
          % (M_DATASETS, N, SIGMA))
    fig, axes = plt.subplots(1, len(MODELS_EXT), figsize=(6 * len(MODELS_EXT), 4.5))
    for ax, (name, fit) in zip(axes, MODELS_EXT.items()):   # 1 กราฟต่อ 1 โมเดล
        r, h, cv, true = run_trials(fit, N, SIGMA, HOLDOUT_RATIO, K_FOLD, rng, M_DATASETS)
        print(f"\n-- model = {name} --   (E_out จริงเฉลี่ย = {np.mean(true):.4f})")
        print(f"{'method':<10} | {'bias':>9} {'variance':>9} {'MSE':>9} {'bias^2+var':>10}")
        print("-" * 54)
        for est, m, c in zip((r, h, cv), METHODS, COLORS):  # วนแต่ละวิธี
            b, v, mse = stats_vs_true(est, true)   # คำนวณสถิติเทียบ E_out จริง
            print(f"{m:<10} | {b:9.4f} {v:9.5f} {mse:9.5f} {b**2 + v:10.5f}")
            ax.hist(est, bins=40, alpha=0.5, color=c, label=m)   # การกระจายค่าประมาณของวิธีนี้
        ax.axvline(np.mean(true), color="k", ls="--", lw=2, label="true E_out")  # เส้นค่าจริง
        ax.set_title(f"model = {name}")            # ชื่อกราฟบอกโมเดล
        ax.set_xlabel("estimated E_out")           # ป้ายแกน x (อังกฤษ: ฟอนต์ไม่มีไทย)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("frequency (datasets)")     # ป้ายแกน y (ใส่แค่กราฟซ้าย)
    plt.tight_layout()
    plt.savefig("estimator_bias_variance_poly3.png", dpi=120)
    print("\nบันทึกรูป estimator_bias_variance_poly3.png แล้ว")


# ======================= (c) เทียบ linear vs poly3: ratio / k =======================

def part_c(rng):
    """เปลี่ยนสัดส่วน holdout และจำนวน fold k แล้วเทียบ variance ระหว่าง linear กับ poly3"""
    print("\n=== (c) variance เทียบ ratio/k : linear vs poly3 ===")
    compare = {"linear": fit_linear, "poly3": fit_poly3}   # 2 โมเดลที่เอามาเทียบกัน
    ratios = [0.2, 0.3, 0.5, 0.7]                  # สัดส่วน test ของ holdout ที่กวาด
    ks = [2, 5, 10, N]                             # จำนวน fold ที่กวาด (k=N คือ LOOCV)
    hold_var, kf_var = {}, {}                      # เก็บ variance ของแต่ละโมเดลไว้วาดกราฟ

    for name, fit in compare.items():              # วนทีละโมเดล (linear, poly3)
        print(f"\n[{name}] holdout (variance ตาม ratio) | k-fold (variance ตาม k)")
        hv, kv = [], []                            # variance list ของ holdout และ kfold
        for ratio in ratios:                       # กวาดสัดส่วน test
            _, h, _, true = run_trials(fit, N, SIGMA, ratio, K_FOLD, rng, M_DATASETS)
            hv.append(stats_vs_true(h, true)[1])   # เก็บเฉพาะ variance (ดัชนี 1)
        for k in ks:                               # กวาดจำนวน fold
            _, _, cv, true = run_trials(fit, N, SIGMA, HOLDOUT_RATIO, k, rng, M_DATASETS)
            kv.append(stats_vs_true(cv, true)[1])
        hold_var[name], kf_var[name] = hv, kv
        # พิมพ์คู่ ratio->var และ k->var ในบรรทัดเดียวเทียบกันได้ง่าย
        print("  holdout: " + " ".join(f"r={r}:{v:.4f}" for r, v in zip(ratios, hv)))
        print("  k-fold : " + " ".join(f"k={k}:{v:.4f}" for k, v in zip(ks, kv)))

    # วาดกราฟเทียบ 2 โมเดล: variance ของ holdout (ตาม ratio) และ k-fold (ตาม k)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in compare:                           # เส้นละ 1 โมเดล
        ax[0].plot(ratios, hold_var[name], "o-", label=name)
        ax[1].plot([str(k) for k in ks], kf_var[name], "o-", label=name)
    ax[0].set_title("holdout: variance vs test ratio"); ax[0].set_xlabel("test ratio")
    ax[1].set_title("k-fold: variance vs folds"); ax[1].set_xlabel("k (folds)")
    for a in ax:
        a.set_yscale("log")                        # log scale: poly3 variance ช่วงกว้างมาก
        a.set_ylabel("variance of estimate (log)"); a.legend()
    plt.tight_layout()
    plt.savefig("split_variance_poly3.png", dpi=120)
    print("\nสรุป: poly3 มี variance ของค่าประมาณสูงกว่า linear ทุกการแบ่ง (ยืดหยุ่นสูงกว่า)")
    print("บันทึกรูป split_variance_poly3.png แล้ว")


# ======================= (d) เทียบ linear vs poly3: n / sigma =======================

def part_d(rng):
    """ปรับ n และ sigma แล้วดูผลต่อ bias/variance เทียบ linear กับ poly3"""
    print("\n=== (d) ผลของ n และ sigma : linear vs poly3 ===")
    compare = {"linear": fit_linear, "poly3": fit_poly3}   # 2 โมเดลที่เทียบกัน
    ns = [10, 20, 50, 100]                         # ค่า n ที่กวาด
    sigmas = [0.1, 0.3, 0.5]                       # ค่า sigma ที่กวาด
    kf_var_by_n = {name: [] for name in compare}   # variance ของ kfold ตาม n (ที่ sigma=SIGMA) ต่อโมเดล

    for name, fit in compare.items():              # วนทีละโมเดล
        print(f"\n-- model = {name} --   (resub bias / kfold variance)")
        print(f"{'n':>4} {'sigma':>6} | {'resub bias':>10} | {'kfold var':>9}")
        print("-" * 38)
        for n in ns:                               # วนขนาดข้อมูล
            for sigma in sigmas:                   # วนระดับ noise
                r, _, cv, true = run_trials(fit, n, sigma, HOLDOUT_RATIO, K_FOLD, rng, M_DATASETS)
                rb = stats_vs_true(r, true)[0]     # bias ของ resub (ดูความมองโลกแง่ดี)
                kv = np.var(cv)                    # variance ของ kfold
                if sigma == SIGMA:                 # เก็บไว้วาดกราฟเฉพาะ sigma ค่ากลาง
                    kf_var_by_n[name].append(kv)
                print(f"{n:>4} {sigma:>6.2f} | {rb:>10.4f} | {kv:>9.4f}")

    # วาดกราฟ: variance ของ kfold ลดลงตาม n เทียบ linear vs poly3 (ที่ sigma=SIGMA)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in compare:                           # เส้นละ 1 โมเดล
        ax.plot(ns, kf_var_by_n[name], "o-", label=name)
    ax.set_yscale("log")                           # log scale: poly3 ที่ n น้อยพุ่งสูงมหาศาล
    ax.set_title(f"k-fold variance vs n (sigma={SIGMA})")
    ax.set_xlabel("n (sample size)"); ax.set_ylabel("variance of estimate (log)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("n_sigma_effect_poly3.png", dpi=120)
    print("\nสรุป: n น้อย poly3 overfit หนัก (resub bias ติดลบมาก + variance สูงกว่า linear ชัด)")
    print("      พอ n มากขึ้นทั้งคู่ดีขึ้น แต่ poly3 ยังแปรปรวนกว่า linear เสมอ")
    print("หมายเหตุ: ที่ n=10 variance ของ poly3 พุ่งสูงมหาศาล (ไม่ใช่บั๊ก) — เพราะพหุนามดีกรี 3")
    print("         fit บนจุดฝึกไม่กี่จุดแล้วเหวี่ยงหลุดกรอบบนช่วง [-1,1] = อาการ overfit สุดขั้ว")
    print("บันทึกรูป n_sigma_effect_poly3.png แล้ว")


def main():
    rng = np.random.default_rng(0)                 # ตัวสุ่มของไฟล์ศึกษาเพิ่มเติม (seed แยกต่างหาก)
    part_a(rng)                                    # (a) ประมาณบนข้อมูลชุดเดียว
    part_b(rng)                                    # (b) bias/variance/MSE
    part_c(rng)                                    # (c) เทียบ ratio/k ระหว่าง linear vs poly3
    part_d(rng)                                    # (d) เทียบ n/sigma ระหว่าง linear vs poly3


if __name__ == "__main__":                         # รันเฉพาะเมื่อสั่งไฟล์นี้ตรง ๆ
    main()
