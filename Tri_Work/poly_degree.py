# -*- coding: utf-8 -*-
"""
งานที่สาม: การเลือกดีกรีของพหุนาม และพฤติกรรม overfitting ด้วยการจำลอง (simulation)

ฟังก์ชันเป้าหมายจริง f(x)=sin(pi*x), สุ่ม x ~ U[-1,1] จำนวน n จุด, บวก noise ~ N(0,sigma^2)
แบบจำลอง = พหุนามดีกรี d (d = 0, 1, ..., D)  (ดึง fit_poly/predict/TARGET จาก models.py)
ประเมิน error 2 แบบ: resubstitution (E_in) และ k-fold cross-validation (E_cv)

หัวข้อในไฟล์นี้:
  (a) ข้อมูล 1 ชุด: หา E_in และ E_cv ของทุกดีกรี วาดเทียบดีกรี + เทียบ E_out จริง หา CV ต่ำสุด
  (b) ทำซ้ำหลายชุด: ค่ากลาง E_in, E_cv, E_out จริง รายดีกรี -> เลือกดีกรีด้วย train vs CV
  (c) ปรับ n และ sigma -> ผลต่อดีกรีที่ดีที่สุด และความรุนแรงของ overfitting
  (d) ขนาดสัมประสิทธิ์สูงสุด |w| เทียบดีกรี -> เชื่อมโยงกับการเกิด overfitting
"""

import sys                                          # ใช้ปรับ encoding ของ output บน Windows
import numpy as np                                 # ไลบรารีคำนวณเชิงตัวเลข
import matplotlib.pyplot as plt                    # ไลบรารีวาดกราฟ
from models import TARGET, fit_poly, predict       # ดึงเป้าหมาย/ฟิตพหุนาม/ทำนาย จาก models.py

# คอนโซล Windows (cp1252) พิมพ์ภาษาไทยไม่ได้ -> บังคับใช้ UTF-8 ให้ตารางแสดงผลถูกต้อง
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------- ค่าคงที่รวมบนสุด (สไตล์ C: ตัวแปรที่ใช้ซ้ำ/ใช้เยอะอยู่ที่เดียว) ----------
SEED = 42                                          # seed ตัวสุ่ม (ผลซ้ำได้ทุกครั้ง)
N = 10                                             # จำนวนตัวอย่าง n เริ่มต้น
SIGMA = 0.3                                         # ค่าเบี่ยงเบนมาตรฐานของ noise เริ่มต้น
MAXDEG = 10                                         # ดีกรีสูงสุด D (จะกวาด d = 0..D)
K_FOLD = 10                                          # จำนวน fold ของ cross-validation
M_DATASETS = 300                                    # จำนวนชุดข้อมูลที่ทำซ้ำ (Monte Carlo)
GRID = np.linspace(-1, 1, 1001)                    # กริด x ถี่ ๆ ไว้คำนวณ E_out จริง


# ======================= ฟังก์ชันย่อยหลัก (ใช้ซ้ำทุกหัวข้อ) =======================

def make_data(n, sigma, rng):
    """สุ่มชุดข้อมูล 1 ชุด: x ~ U[-1,1] จำนวน n จุด, y = f(x) + noise"""
    x = rng.uniform(-1, 1, n)                      # สุ่ม x จากการแจกแจงเอกรูปในช่วง [-1,1]
    y = TARGET(x) + sigma * rng.standard_normal(n) # ค่าจริง sin(pi*x) บวก noise แบบปรกติ
    return x, y


def mse(coef, x, y):
    """error กำลังสองเฉลี่ยของพหุนาม coef บนจุด (x, y)"""
    return np.mean((predict(coef, x) - y) ** 2)


def true_eout(coef, sigma):
    """E_out จริงของพหุนามที่ฟิตแล้ว = ค่าคาดหวัง error บนจุดทดสอบใหม่ (มี noise)
       = E_x[(g(x)-f(x))^2] (ประมาณด้วยกริดถี่) + sigma^2 (noise floor ที่ลดไม่ได้)"""
    return np.mean((predict(coef, GRID) - TARGET(GRID)) ** 2) + sigma ** 2


def est_resub(x, y, d):
    """resubstitution (E_in): ฟิตดีกรี d บนข้อมูลทั้งหมด แล้ววัด error บนจุดเดิม"""
    return mse(fit_poly(x, y, d), x, y)            # error บนชุดเดิม (มักต่ำกว่า E_out จริง)


def est_kfold(x, y, d, k, rng):
    """k-fold CV (E_cv): สับข้อมูลแล้วแบ่งเป็น k ก้อน วนให้แต่ละก้อนเป็น test ทีละครั้ง
       ฟิตดีกรี d บนก้อนที่เหลือ วัด error บนก้อนที่กันไว้ แล้วเฉลี่ยทุกก้อน"""
    folds = np.array_split(rng.permutation(len(x)), k)   # สับดัชนีแล้วแบ่งเป็น k ก้อน
    errs = []                                      # เก็บ error ของแต่ละ fold ไว้เฉลี่ย
    for i in range(k):                             # วนเลือก fold ที่ i เป็น test
        test = folds[i]                            # ก้อนที่ i = ชุดทดสอบ
        train = np.concatenate([folds[j] for j in range(k) if j != i])  # ก้อนที่เหลือ = ชุดฝึก
        errs.append(mse(fit_poly(x[train], y[train], d), x[test], y[test]))  # error บน fold ทดสอบ
    return np.mean(errs)                           # เฉลี่ย error เหนือทุก fold


def errors_by_degree(x, y, sigma, k, rng, D):
    """สำหรับข้อมูล 'ชุดเดียว': วน d = 0..D คืน 3 อาเรย์ (E_in, E_cv, E_out จริง) รายดีกรี"""
    e_in, e_cv, e_out = [], [], []                 # ลิสต์สะสมค่าของแต่ละดีกรี
    for d in range(D + 1):                         # วนทุกดีกรีตั้งแต่ 0 ถึง D
        e_in.append(est_resub(x, y, d))            # training error (resubstitution)
        e_cv.append(est_kfold(x, y, d, k, rng))    # cross-validation error
        e_out.append(true_eout(fit_poly(x, y, d), sigma))  # E_out จริงของโมเดลเต็มชุด
    return map(np.array, (e_in, e_cv, e_out))      # แปลงเป็น numpy array ทั้ง 3 ตัว


# ======================= หัวข้อ (a) =======================

def part_a(rng):
    """ข้อมูล 1 ชุด: หา E_in และ E_cv ของทุกดีกรี วาดเทียบดีกรี + เทียบ E_out จริง"""
    print("\n=== (a) error รายดีกรีบนข้อมูล 1 ชุด (n=%d, sigma=%.2f, D=%d) ==="
          % (N, SIGMA, MAXDEG))
    x, y = make_data(N, SIGMA, rng)                # สุ่มข้อมูล 1 ชุด
    e_in, e_cv, e_out = errors_by_degree(x, y, SIGMA, K_FOLD, rng, MAXDEG)
    best = int(np.argmin(e_cv))                    # ดีกรีที่ CV ต่ำสุด (ตัวเลือกของ CV)

    print(f"{'d':>3} | {'E_in':>9} {'E_cv':>9} {'E_out(จริง)':>11}")
    print("-" * 40)
    for d in range(MAXDEG + 1):                    # พิมพ์ทีละดีกรี
        mark = "  <- CV ต่ำสุด" if d == best else ""
        print(f"{d:>3} | {e_in[d]:9.4f} {e_cv[d]:9.4f} {e_out[d]:11.4f}{mark}")
    print(f"สังเกต: E_in ลดลงเรื่อย ๆ ตามดีกรี (overfit) แต่ E_cv เป็นรูปตัว U -> CV เลือก d={best}")

    # --- วาดกราฟ error เทียบดีกรี (ป้ายอังกฤษ เพราะฟอนต์ปริยายไม่มีตัวไทย) ---
    degs = np.arange(MAXDEG + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(degs, e_in, "o-", color="tab:red", label="E_in (train / resub)")
    ax.plot(degs, e_cv, "s-", color="tab:blue", label="E_cv (k-fold)")
    ax.plot(degs, e_out, "^--", color="k", label="true E_out")
    ax.axvline(best, color="tab:blue", ls=":", lw=1.5)     # เส้นดีกรีที่ CV ต่ำสุด
    ax.scatter([best], [e_cv[best]], s=120, facecolors="none", edgecolors="tab:blue", zorder=5)
    ax.set_yscale("log")                           # log ช่วยเห็น E_in ที่ต่ำมากตอนดีกรีสูง
    ax.set_title("(a) error vs polynomial degree (single dataset)")
    ax.set_xlabel("degree d"); ax.set_ylabel("MSE (log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("degree_curves.png", dpi=120)
    print("บันทึกรูป degree_curves.png แล้ว")


# ======================= หัวข้อ (b) =======================

def mean_errors(n, sigma, k, rng, D, M):
    """ทำซ้ำ M ชุดข้อมูล เฉลี่ย E_in, E_cv, E_out จริง รายดีกรี -> คืน 3 อาเรย์ (ยาว D+1)"""
    acc = np.zeros((3, D + 1))                     # ตัวสะสมผลรวมของ 3 ชนิด error รายดีกรี
    for _ in range(M):                             # วนทำซ้ำ M ชุด
        x, y = make_data(n, sigma, rng)            # สุ่มชุดข้อมูลใหม่
        for i, arr in enumerate(errors_by_degree(x, y, sigma, k, rng, D)):
            acc[i] += arr                          # บวกสะสมทีละชนิด (E_in/E_cv/E_out)
    return acc / M                                 # หารด้วย M -> ค่ากลาง


def part_b(rng):
    """ทำซ้ำ M ชุด: ค่ากลาง error รายดีกรี -> เทียบการเลือกดีกรีด้วย train vs CV"""
    print("\n=== (b) ค่ากลาง error รายดีกรี (ทำซ้ำ %d ชุด, n=%d, sigma=%.2f) ==="
          % (M_DATASETS, N, SIGMA))
    e_in, e_cv, e_out = mean_errors(N, SIGMA, K_FOLD, rng, MAXDEG, M_DATASETS)
    d_train = int(np.argmin(e_in))                 # ดีกรีที่ training error ต่ำสุด
    d_cv = int(np.argmin(e_cv))                    # ดีกรีที่ CV ต่ำสุด
    d_out = int(np.argmin(e_out))                  # ดีกรีที่ E_out จริงต่ำสุด (คำตอบในอุดมคติ)

    print(f"{'d':>3} | {'E_in':>9} {'E_cv':>9} {'E_out(จริง)':>11}")
    print("-" * 40)
    for d in range(MAXDEG + 1):
        print(f"{d:>3} | {e_in[d]:9.4f} {e_cv[d]:9.4f} {e_out[d]:11.4f}")
    print(f"\nเลือกดีกรีด้วย training error -> d={d_train}  (มัก = D เพราะ E_in ลดลงตามดีกรีเสมอ = overfit)")
    print(f"เลือกดีกรีด้วย CV            -> d={d_cv}   เทียบดีกรีดีที่สุดจริง (E_out ต่ำสุด) -> d={d_out}")
    print(f"|E_cv - E_out| เฉลี่ยทุกดีกรี = {np.mean(np.abs(e_cv - e_out)):.4f}  (ยิ่งน้อย = CV ยิ่งแทน E_out ได้ดี)")

    # --- วาดกราฟค่ากลาง 3 เส้นเทียบดีกรี ---
    degs = np.arange(MAXDEG + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(degs, e_in, "o-", color="tab:red", label="mean E_in")
    ax.plot(degs, e_cv, "s-", color="tab:blue", label="mean E_cv")
    ax.plot(degs, e_out, "^--", color="k", label="mean true E_out")
    ax.axvline(d_cv, color="tab:blue", ls=":", lw=1.5, label=f"CV pick d={d_cv}")
    ax.axvline(d_out, color="k", ls=":", lw=1.5, label=f"best d={d_out}")
    ax.set_yscale("log")
    ax.set_title(f"(b) mean error vs degree ({M_DATASETS} datasets)")
    ax.set_xlabel("degree d"); ax.set_ylabel("mean MSE (log scale)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("mean_curves.png", dpi=120)
    print("บันทึกรูป mean_curves.png แล้ว")


# ======================= หัวข้อ (c) =======================

def part_c(rng):
    """ปรับ n และ sigma -> ผลต่อดีกรีที่ดีที่สุด (CV) และความรุนแรงของ overfitting"""
    print("\n=== (c) ผลของ n และ sigma ต่อดีกรีที่ดีที่สุด และ overfitting ===")
    ns = [10, 20, 40, 80]                          # ค่า n ที่จะกวาด
    sigmas = [0.0, 0.2, 0.5]                        # ค่า sigma ที่จะกวาด
    # overfit gap = ช่องว่าง E_out - E_in ที่ดีกรีสูงสุด D (ยิ่งกว้าง = overfit ยิ่งแรง)
    best_by_sig = {s: [] for s in sigmas}          # ดีกรีที่ CV ต่ำสุด แยกตาม sigma
    gap_by_sig = {s: [] for s in sigmas}           # overfit gap แยกตาม sigma

    print(f"{'n':>4} {'sigma':>6} | {'best_d(CV)':>10} {'overfit_gap':>12}")
    print("-" * 40)
    for sigma in sigmas:                           # วนแต่ละระดับ noise
        for n in ns:                               # วนแต่ละขนาดข้อมูล
            e_in, e_cv, e_out = mean_errors(n, sigma, K_FOLD, rng, MAXDEG, M_DATASETS)
            best = int(np.argmin(e_cv))            # ดีกรีที่ CV เฉลี่ยต่ำสุด
            gap = e_out[MAXDEG] - e_in[MAXDEG]     # ช่องว่างที่ดีกรีสูงสุด = ความรุนแรง overfit
            best_by_sig[sigma].append(best)
            gap_by_sig[sigma].append(gap)
            print(f"{n:>4} {sigma:>6.2f} | {best:>10} {gap:>12.4f}")

    # --- วาด 2 subplot: best degree vs n และ overfit gap vs n (เส้นละ sigma) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for sigma in sigmas:
        ax[0].plot(ns, best_by_sig[sigma], "o-", label=f"sigma={sigma}")
        ax[1].plot(ns, gap_by_sig[sigma], "o-", label=f"sigma={sigma}")
    ax[0].set_title("(c) best degree (min CV) vs n")
    ax[0].set_xlabel("n (sample size)"); ax[0].set_ylabel("best degree d"); ax[0].legend()
    ax[1].set_title("(c) overfitting gap (E_out - E_in at d=D) vs n")
    ax[1].set_xlabel("n (sample size)"); ax[1].set_ylabel("overfit gap"); ax[1].legend()
    ax[1].set_yscale("log")                         # gap เหวี่ยงกว้าง -> log อ่านง่ายกว่า
    plt.tight_layout()
    plt.savefig("n_sigma_effect.png", dpi=120)
    print("\nสรุป: n มากขึ้น -> รับดีกรีสูงได้/overfit ลดลง;  sigma มากขึ้น -> best degree ต่ำลง/overfit แรงขึ้น")
    print("บันทึกรูป n_sigma_effect.png แล้ว")


# ======================= หัวข้อ (d) =======================

def part_d(rng):
    """ขนาดสัมประสิทธิ์สูงสุด |w| เทียบดีกรี -> เชื่อมโยงกับการเกิด overfitting"""
    print("\n=== (d) ขนาดสัมประสิทธิ์สูงสุด |w| เทียบดีกรี (เฉลี่ย %d ชุด, n=%d, sigma=%.2f) ==="
          % (M_DATASETS, N, SIGMA))
    maxw = np.zeros(MAXDEG + 1)                     # ค่าเฉลี่ย max|w| รายดีกรี
    e_cv = np.zeros(MAXDEG + 1)                     # ค่าเฉลี่ย E_cv รายดีกรี (ไว้วางคู่กัน)
    for _ in range(M_DATASETS):                    # ทำซ้ำหลายชุดแล้วเฉลี่ย
        x, y = make_data(N, SIGMA, rng)
        for d in range(MAXDEG + 1):
            coef = fit_poly(x, y, d)               # ฟิตพหุนามดีกรี d
            maxw[d] += np.max(np.abs(coef))        # สัมประสิทธิ์ที่ใหญ่สุด (ขนาด)
            e_cv[d] += est_kfold(x, y, d, K_FOLD, rng)
    maxw /= M_DATASETS; e_cv /= M_DATASETS

    print(f"{'d':>3} | {'max|w|':>12} {'E_cv':>9}")
    print("-" * 30)
    for d in range(MAXDEG + 1):
        print(f"{d:>3} | {maxw[d]:12.3f} {e_cv[d]:9.4f}")
    print("สังเกต: ดีกรีสูง -> max|w| โตแบบระเบิด (สัมประสิทธิ์บวก/ลบใหญ่หักล้างกันเพื่อลากผ่านทุกจุด)")

    # --- วาดแยก 2 กราฟข้างกัน: (ซ้าย) max|w|  (ขวา) E_cv  ทั้งคู่แกน log ให้เห็นการไต่ขึ้น ---
    degs = np.arange(MAXDEG + 1)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    axL.plot(degs, maxw, "o-", color="tab:purple")   # ซ้าย: ขนาดสัมประสิทธิ์
    axL.set_yscale("log")
    axL.set_xlabel("degree d"); axL.set_ylabel("max |w| (log scale)")
    axL.set_title("(d1) coefficient magnitude vs degree")
    axL.grid(True, ls="--", alpha=0.4)

    axR.plot(degs, e_cv, "s--", color="tab:blue")     # ขวา: cross-validation error
    axR.set_yscale("log")
    axR.set_xlabel("degree d"); axR.set_ylabel("E_cv (log scale)")
    axR.set_title("(d2) cross-validation error vs degree")
    axR.grid(True, ls="--", alpha=0.4)

    fig.suptitle("(d) exploding |w| tracks exploding E_cv  (overfitting signal)")
    plt.tight_layout()
    plt.savefig("coef_growth.png", dpi=120)
    print("สรุป: |w| ที่พุ่งสูง = สัญญาณ overfitting; regularization คุมขนาด w เพื่อกันอาการนี้")
    print("บันทึกรูป coef_growth.png แล้ว")


def main():
    rng = np.random.default_rng(SEED)              # ตัวสุ่มกลาง ใช้ร่วมทุกหัวข้อ (ผลซ้ำได้)
    part_a(rng)                                    # (a) error รายดีกรีบนข้อมูลชุดเดียว
    part_b(rng)                                    # (b) ค่ากลางจากการทำซ้ำ -> train vs CV
    part_c(rng)                                    # (c) ปรับ n / sigma
    part_d(rng)                                    # (d) ขนาดสัมประสิทธิ์ |w|


if __name__ == "__main__":                         # รันเฉพาะเมื่อสั่งไฟล์นี้ตรง ๆ
    main()
