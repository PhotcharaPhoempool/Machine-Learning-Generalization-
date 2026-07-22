# -*- coding: utf-8 -*-
"""
งานที่สอง: ความเอนเอียง (bias) และความแปรปรวน (variance) ของ *ตัวประมาณค่า E_out*
3 แบบ ได้แก่ 1) resubstitution 2) holdout 3) k-fold cross-validation

ฟังก์ชันเป้าหมาย f(x)=sin(pi*x), สุ่ม x ~ U[-1,1] จำนวน n จุด, บวก noise ~ N(0,sigma^2)
แบบจำลอง 2 แบบ: constant, linear  (ดึงมาจากงานแรกผ่าน models.py)

หัวข้อในไฟล์นี้:
  (a) ประมาณ E_out ด้วย 3 วิธีบนข้อมูล 1 ชุด แล้วเทียบกับ E_out จริง
  (b) ทำซ้ำหลายชุด -> bias, variance, MSE ของแต่ละวิธี
  (c) เปลี่ยนสัดส่วน holdout และจำนวน fold k -> เทียบความแปรปรวน
  (d) ปรับ n และ sigma -> สังเกตผลต่อ bias/variance
"""

import sys                                          # ใช้ปรับ encoding ของ output บน Windows
import numpy as np                                 # ไลบรารีคำนวณเชิงตัวเลข
import matplotlib.pyplot as plt                    # ไลบรารีวาดกราฟ
from models import MODELS, TARGET, predict         # ดึงแบบจำลอง/เป้าหมาย/ทำนาย จากงานแรก

# คอนโซล Windows (cp1252) พิมพ์ภาษาไทยไม่ได้ -> บังคับใช้ UTF-8 ให้ตารางแสดงผลถูกต้อง
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------- ค่าคงที่รวมบนสุด (ประกาศแบบ C: ตัวแปรที่ใช้ซ้ำ/ใช้เยอะอยู่ที่เดียว) ----------
SEED = 42                                          # seed ตัวสุ่ม (ผลซ้ำได้ทุกครั้ง)
N = 2                                             # จำนวนตัวอย่าง n เริ่มต้น
SIGMA = 0                                        # ค่าเบี่ยงเบนมาตรฐานของ noise เริ่มต้น
M_DATASETS = 3000                                  # จำนวนชุดข้อมูลที่ทำซ้ำ (Monte Carlo)
GRID = np.linspace(-1, 1, 1001)                    # กริด x ไว้คำนวณ E_out จริง (ส่วน deterministic)
HOLDOUT_RATIO = 0.3                                # สัดส่วนข้อมูลที่กันไว้เป็น test ของ holdout เริ่มต้น
K_FOLD = 5                                         # จำนวน fold ของ cross-validation เริ่มต้น


# ======================= ฟังก์ชันย่อยหลัก =======================

def make_data(n, sigma, rng):
    """สุ่มชุดข้อมูล 1 ชุด: x ~ U[-1,1] จำนวน n จุด, y = f(x) + noise"""
    x = rng.uniform(-1, 1, n)                      # สุ่ม x จากการแจกแจงเอกรูปในช่วง [-1,1]
    y = TARGET(x) + sigma * rng.standard_normal(n) # ค่าจริง sin(pi*x) บวก noise แบบปรกติ
    return x, y


def true_eout(theta, sigma):
    """E_out จริงของโมเดลที่เทรนแล้ว (theta): ค่าคาดหวังของ error บนจุดทดสอบใหม่ที่มี noise
       = E_x[(g(x)-f(x))^2] (ประมาณด้วยกริดถี่) + sigma^2 (ส่วน noise ที่ลดไม่ได้)"""
    bias_part = np.mean((predict(theta, GRID) - TARGET(GRID)) ** 2)  # error เชิงเฉลี่ยเทียบ f จริง
    return bias_part + sigma ** 2                  # บวก noise floor -> เป็น test error จริง


def est_resub(x, y, fit):
    """Resubstitution: เทรนบนข้อมูลทั้ง n จุด แล้ววัด error บน n จุดเดิม (in-sample)"""
    theta = fit(x, y)                              # เทรนโมเดลบนข้อมูลทั้งหมด
    return np.mean((predict(theta, x) - y) ** 2)   # error บนชุดเดิม (มักต่ำกว่า E_out จริง)


def est_holdout(x, y, fit, ratio, rng):
    """Holdout: สุ่มแบ่งข้อมูลเป็น train/test ตาม ratio (สัดส่วน test),
       เทรนบน train แล้ววัด error บน test"""
    n = len(x)                                     # จำนวนข้อมูลทั้งหมด
    idx = rng.permutation(n)                       # สุ่มสับลำดับดัชนีก่อนแบ่ง (ไม่ให้เอนเอียง)
    n_test = max(1, int(round(ratio * n)))         # จำนวนจุด test (อย่างน้อย 1 จุด)
    test, train = idx[:n_test], idx[n_test:]       # แบ่งดัชนีเป็นส่วน test และ train
    theta = fit(x[train], y[train])                # เทรนบนเฉพาะส่วน train
    return np.mean((predict(theta, x[test]) - y[test]) ** 2)  # error บนส่วน test ที่ไม่เคยเห็น


def est_kfold(x, y, fit, k, rng):
    """k-fold CV: สับข้อมูลแล้วแบ่งเป็น k ก้อน วนให้แต่ละก้อนเป็น test ทีละครั้ง
       เทรนบนก้อนที่เหลือ วัด error บนก้อนที่กันไว้ แล้วเฉลี่ยทุกก้อน"""
    n = len(x)                                     # จำนวนข้อมูลทั้งหมด
    folds = np.array_split(rng.permutation(n), k)  # สับดัชนีแล้วแบ่งเป็น k ก้อน (ขนาดใกล้เคียงกัน)
    errs = []                                      # เก็บ error ของแต่ละ fold ไว้เฉลี่ย
    for i in range(k):                             # วนเลือก fold ที่ i เป็น test
        test = folds[i]                            # ก้อนที่ i = ชุดทดสอบ
        train = np.concatenate([folds[j] for j in range(k) if j != i])  # ก้อนที่เหลือ = ชุดฝึก
        theta = fit(x[train], y[train])            # เทรนบนชุดฝึกของรอบนี้
        errs.append(np.mean((predict(theta, x[test]) - y[test]) ** 2))  # error บน fold ทดสอบ
    return np.mean(errs)                           # เฉลี่ย error เหนือทุก fold


def run_trials(fit, n, sigma, ratio, k, rng, M):
    """ทำซ้ำ M ชุดข้อมูล: แต่ละชุดเก็บค่าประมาณจาก 3 วิธี + E_out จริงของโมเดลเต็มชุด
       คืน 4 อาเรย์ (ยาว M): resub, holdout, kfold, true"""
    resub, hold, kf, true = [], [], [], []         # ลิสต์สะสมผลของแต่ละชุดข้อมูล
    for _ in range(M):                             # วนทำซ้ำ M รอบ
        x, y = make_data(n, sigma, rng)            # สุ่มชุดข้อมูลใหม่
        resub.append(est_resub(x, y, fit))         # ค่าประมาณจาก resubstitution
        hold.append(est_holdout(x, y, fit, ratio, rng))   # ค่าประมาณจาก holdout
        kf.append(est_kfold(x, y, fit, k, rng))    # ค่าประมาณจาก k-fold CV
        true.append(true_eout(fit(x, y), sigma))   # E_out จริงของโมเดลที่เทรนบนข้อมูลเต็มชุด
    return map(np.array, (resub, hold, kf, true))  # แปลงเป็น numpy array ทั้ง 4 ตัว


def stats_vs_true(est, true):
    """คำนวณสถิติของตัวประมาณ 1 วิธี เทียบกับ E_out จริง:
       bias = ค่าเฉลี่ยส่วนต่าง, var = ความแปรปรวนของค่าประมาณ, MSE = ค่าเฉลี่ยส่วนต่างกำลังสอง"""
    bias = np.mean(est) - np.mean(true)            # ความเอนเอียง (บวก=สูงไป, ลบ=ต่ำไป/มองโลกแง่ดี)
    var = np.var(est)                              # ความแปรปรวนของค่าที่ประมาณได้
    mse = np.mean((est - true) ** 2)               # MSE เทียบ E_out จริง (โดยประมาณ = bias^2 + var)
    return bias, var, mse


# ======================= หัวข้อ (a) =======================

def part_a(rng):
    """ประมาณ E_out ด้วย 3 วิธีบนข้อมูล 'ชุดเดียว' แล้วเทียบกับ E_out จริง"""
    print("\n=== (a) ประมาณ E_out บนข้อมูล 1 ชุด (n=%d, sigma=%.2f) ===" % (N, SIGMA))
    print(f"{'model':<10} | {'resub':>8} {'holdout':>8} {'kfold':>8} | {'E_out(จริง)':>10}")
    print("-" * 54)
    for name, fit in MODELS.items():               # วนแต่ละแบบจำลอง
        x, y = make_data(N, SIGMA, rng)            # สุ่มข้อมูล 1 ชุดสำหรับโมเดลนี้
        r = est_resub(x, y, fit)                   # ค่าประมาณ resubstitution
        h = est_holdout(x, y, fit, HOLDOUT_RATIO, rng)   # ค่าประมาณ holdout
        cv = est_kfold(x, y, fit, K_FOLD, rng)     # ค่าประมาณ k-fold
        t = true_eout(fit(x, y), SIGMA)            # E_out จริงของโมเดลเต็มชุด
        print(f"{name:<10} | {r:8.4f} {h:8.4f} {cv:8.4f} | {t:10.4f}")
    print("สังเกต: resub มักต่ำกว่า E_out จริง (มองโลกแง่ดี), holdout/kfold ใกล้จริงกว่า")


# ======================= หัวข้อ (b) =======================

def part_b(rng):
    """ทำซ้ำ M ชุด -> หา bias, variance, MSE ของตัวประมาณแต่ละวิธี + วาด histogram"""
    print("\n=== (b) bias / variance / MSE ของแต่ละวิธี (ทำซ้ำ %d ชุด, n=%d, sigma=%.2f) ==="
          % (M_DATASETS, N, SIGMA))
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.5))
    methods = ["resub", "holdout", "kfold"]        # ชื่อวิธีไว้วนพิมพ์/วาด
    colors = ["tab:red", "tab:green", "tab:blue"]  # สีประจำแต่ละวิธีในกราฟ

    for ax, (name, fit) in zip(axes, MODELS.items()):   # วนแต่ละแบบจำลอง (1 กราฟต่อโมเดล)
        r, h, cv, true = run_trials(fit, N, SIGMA, HOLDOUT_RATIO, K_FOLD, rng, M_DATASETS)
        print(f"\n-- model = {name} --   (E_out จริงเฉลี่ย = {np.mean(true):.4f})")
        print(f"{'method':<10} | {'bias':>9} {'variance':>9} {'MSE':>9} {'bias^2+var':>10}")
        print("-" * 54)
        for est, m, c in zip((r, h, cv), methods, colors):   # วนแต่ละวิธีของโมเดลนี้
            b, v, mse = stats_vs_true(est, true)   # คำนวณสถิติเทียบ E_out จริง
            # พิมพ์ค่า + คอลัมน์ bias^2+var ไว้ตรวจว่าใกล้ MSE (ยืนยันการ decompose ถูก)
            print(f"{m:<10} | {b:9.4f} {v:9.5f} {mse:9.5f} {b**2 + v:10.5f}")
            ax.hist(est, bins=40, alpha=0.5, color=c, label=m)   # การกระจายค่าประมาณของวิธีนี้
        # ป้ายกราฟใช้ภาษาอังกฤษ (ฟอนต์ matplotlib ปริยายไม่มีสระ/พยัญชนะไทย)
        ax.axvline(np.mean(true), color="k", ls="--", lw=2, label="true E_out")  # เส้นค่าจริง
        ax.set_title(f"model = {name}")            # ชื่อกราฟบอกโมเดล
        ax.set_xlabel("estimated E_out")           # ป้ายแกน x
        ax.legend(fontsize=8)                      # คำอธิบายเส้น/แท่ง
    axes[0].set_ylabel("frequency (datasets)")     # ป้ายแกน y (ใส่แค่กราฟซ้าย)
    plt.tight_layout()
    plt.savefig("estimator_bias_variance.png", dpi=120)
    print("\nบันทึกรูป estimator_bias_variance.png แล้ว")


# ======================= หัวข้อ (c) =======================

def part_c(rng):
    """เปลี่ยนสัดส่วน holdout และจำนวน fold k -> เทียบความแปรปรวนของค่าที่ประมาณได้"""
    print("\n=== (c) ผลของสัดส่วน holdout และจำนวน fold k ต่อความแปรปรวน (model=linear) ===")
    fit = MODELS["linear"]                         # เลือก linear เป็นตัวแทน (ยืดหยุ่นกว่า เห็นผลชัด)
    ratios = [0.2, 0.3, 0.5, 0.7]                  # สัดส่วน test ของ holdout ที่จะกวาด
    ks = [2, 5, 10, N]                             # จำนวน fold ที่จะกวาด (k=N คือ LOOCV)

    # --- holdout: กวาด ratio แล้วเก็บ bias/variance ของค่าประมาณ ---
    print("\n[holdout] เทียบสัดส่วน test:")
    print(f"{'ratio':>6} | {'bias':>9} {'variance':>10}")
    print("-" * 30)
    hold_var = []                                  # เก็บ variance ของแต่ละ ratio ไว้วาดกราฟ
    for ratio in ratios:                           # วนแต่ละสัดส่วน test
        _, h, _, true = run_trials(fit, N, SIGMA, ratio, K_FOLD, rng, M_DATASETS)
        b, v, _ = stats_vs_true(h, true)           # bias/var ของ holdout ที่ ratio นี้
        hold_var.append(v)
        print(f"{ratio:6.1f} | {b:9.4f} {v:10.5f}")

    # --- k-fold: กวาด k แล้วเก็บ bias/variance ของค่าประมาณ ---
    print("\n[k-fold] เทียบจำนวน fold (k=%d คือ LOOCV):" % N)
    print(f"{'k':>6} | {'bias':>9} {'variance':>10}")
    print("-" * 30)
    kf_var = []                                    # เก็บ variance ของแต่ละ k ไว้วาดกราฟ
    for k in ks:                                   # วนแต่ละจำนวน fold
        _, _, cv, true = run_trials(fit, N, SIGMA, HOLDOUT_RATIO, k, rng, M_DATASETS)
        b, v, _ = stats_vs_true(cv, true)          # bias/var ของ k-fold ที่ k นี้
        kf_var.append(v)
        print(f"{k:6d} | {b:9.4f} {v:10.5f}")

    # --- วาดกราฟเทียบ variance ของทั้งสองแบบการแบ่ง ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    # ป้ายกราฟใช้ภาษาอังกฤษ (ฟอนต์ปริยายไม่มีตัวอักษรไทย)
    ax[0].plot(ratios, hold_var, "o-", color="tab:green")   # variance ของ holdout ตาม ratio
    ax[0].set_title("holdout: variance vs test ratio")
    ax[0].set_xlabel("test ratio"); ax[0].set_ylabel("variance of estimate")
    ax[1].plot([str(k) for k in ks], kf_var, "o-", color="tab:blue")   # variance ของ k-fold ตาม k
    ax[1].set_title("k-fold: variance vs number of folds")
    ax[1].set_xlabel("k (folds)"); ax[1].set_ylabel("variance of estimate")
    plt.tight_layout()
    plt.savefig("split_variance.png", dpi=120)
    print("\nสรุป: test set ใหญ่ขึ้น (ratio มาก) / k มากขึ้น -> variance ของค่าประมาณมักลดลง")
    print("บันทึกรูป split_variance.png แล้ว")


# ======================= หัวข้อ (d) =======================

def part_d(rng):
    """ปรับ n และ sigma -> สังเกตผลต่อ bias และ variance ของแต่ละวิธี (model=linear)"""
    print("\n=== (d) ผลของ n และ sigma ต่อ bias/variance (model=linear) ===")
    fit = MODELS["linear"]                         # ใช้ linear เป็นตัวแทน
    ns = [10, 20, 50, 100]                         # ค่า n ที่จะกวาด
    sigmas = [0.1, 0.3, 0.5]                       # ค่า sigma ที่จะกวาด
    methods = ["resub", "holdout", "kfold"]

    print(f"{'n':>4} {'sigma':>6} | " +
          " | ".join(f"{m+' b/var':>16}" for m in methods))
    print("-" * 66)
    # เก็บ variance ของ kfold ตาม n (ที่ sigma กลาง) ไว้วาดกราฟสรุป
    kf_var_by_n = {s: [] for s in sigmas}
    for n in ns:                                   # วนแต่ละขนาดข้อมูล
        for sigma in sigmas:                       # วนแต่ละระดับ noise
            r, h, cv, true = run_trials(fit, n, sigma, HOLDOUT_RATIO, K_FOLD, rng, M_DATASETS)
            cells = []                             # เก็บข้อความ bias/var ของแต่ละวิธีไว้พิมพ์เป็นแถว
            for est in (r, h, cv):
                b, v, _ = stats_vs_true(est, true)
                cells.append(f"{b:+.3f}/{v:.4f}")
            kf_var_by_n[sigma].append(np.var(cv))  # บันทึก variance ของ kfold ไว้วาดกราฟ
            print(f"{n:>4} {sigma:>6.2f} | " + " | ".join(f"{c:>16}" for c in cells))

    # --- วาดกราฟ: variance ของ kfold ลดลงอย่างไรเมื่อ n เพิ่ม แยกตาม sigma ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sigma in sigmas:                           # เส้นละ 1 ระดับ noise
        ax.plot(ns, kf_var_by_n[sigma], "o-", label=f"sigma={sigma}")
    # ป้ายกราฟใช้ภาษาอังกฤษ (ฟอนต์ปริยายไม่มีตัวอักษรไทย)
    ax.set_title("k-fold: variance of estimate vs n (per sigma)")
    ax.set_xlabel("n (sample size)"); ax.set_ylabel("variance of estimate")
    ax.legend()
    plt.tight_layout()
    plt.savefig("n_sigma_effect.png", dpi=120)
    print("\nสรุป: n มากขึ้น -> variance ทุกวิธีลดลง; sigma มากขึ้น -> variance เพิ่ม และ resub มองโลกแง่ดีชัดขึ้น")
    print("บันทึกรูป n_sigma_effect.png แล้ว")


def main():
    rng = np.random.default_rng(SEED)              # ตัวสุ่มกลาง ใช้ร่วมทุกหัวข้อ (ผลซ้ำได้)
    part_a(rng)                                    # (a) ประมาณบนข้อมูลชุดเดียว
    part_b(rng)                                    # (b) bias/variance/MSE จากการทำซ้ำ
    part_c(rng)                                    # (c) เปลี่ยน ratio / k เทียบ variance
    part_d(rng)                                    # (d) ปรับ n / sigma


if __name__ == "__main__":                         # รันเฉพาะเมื่อสั่งไฟล์นี้ตรง ๆ
    main()
