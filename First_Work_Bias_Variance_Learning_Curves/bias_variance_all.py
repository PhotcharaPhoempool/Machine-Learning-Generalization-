# -*- coding: utf-8 -*-
"""
รันทั้งงาน: Bias-Variance (ส่วน 1-2) + Learning Curves (ส่วน 3)
ได้รูปทั้งสองใบในครั้งเดียว  (ตรรกะจริงอยู่ใน part1_/part2_ + models.py)
"""

import part1_bias_variance                        # นำเข้าส่วน 1-2 (bias/variance)
import part2_learning_curves                      # นำเข้าส่วน 3 (learning curves)

if __name__ == "__main__":                        # รันเฉพาะเมื่อสั่งไฟล์นี้ตรง ๆ
    part1_bias_variance.main()                    # รันส่วน 1-2 -> ได้ bias_variance.png
    part2_learning_curves.main()                  # รันส่วน 3   -> ได้ learning_curves.png
