#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

test = Image.open('1517186033652.png')
zhi2 = test.convert('L')

zhi2.show()

print(np.array(zhi2))

