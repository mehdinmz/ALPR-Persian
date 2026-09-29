import csv
import os
import random
import time
from PIL import Image
import PIL
import cv2
import imutils
import numpy as np
from tqdm import tqdm

# Characters of Letters and Numbers in Plates
numbers = [str(i) for i in range(0, 10)]

# Fonts and Templates
fonts = ['roya_bold']
t = "C:\\Users\\MEHDI\\Desktop\\iranian-license-plate-recognition\\Dataset\\Templates"
templates = [
    os.path.basename(os.path.splitext(template)[0])
    for template in os.listdir(t)
    if template.endswith('.png')
    and template not in ['tashrifat.png', 'template-sepah.png', 'template-police.png']
]

# Noises
n = "C:\\Users\\MEHDI\\Desktop\\iranian-license-plate-recognition\\Dataset\\Noises"
noises = os.listdir(n)
transformations = ['rotate_right', 'rotate_left', 'zoom_in', 'zoom_out', 'prespective_transform']

# Count of permutations
permutations = 1

def getPlateName(n1, n2, l, n3, n4, n5, c1, c2):
    return f'{n1}{n2}{l}{n3}{n4}{n5}_{c1}{c2}'

def getGlyphAddress(font, glyphName):
    return f'../Glyphs/{font}/{glyphName}_trim.png'

def getNewPlate(template_name, available_letters):
    """
    تولید ساختار کامل پلاک (۸ کاراکتر):
    [n1, n2, letter, n3, n4, n5, city1, city2]
    """
    valid_letters = available_letters.copy()
    
    # کنترل محدودیت حرف 'ع' (EIN / EA): فقط برای قالب‌های عمومی/زرد
    is_public_template = 'public' in template_name.lower() or 'yellow' in template_name.lower()
    
    if not is_public_template:
        valid_letters = [l for l in valid_letters if l not in ['EIN', 'EA', 'EIN_trim']]

    selected_letter = random.choice(valid_letters) if valid_letters else random.choice(available_letters)

    return [
        random.choice(numbers),  # n1
        random.choice(numbers),  # n2
        selected_letter,         # letter
        random.choice(numbers),  # n3
        random.choice(numbers),  # n4
        random.choice(numbers),  # n5
        random.choice(numbers),  # city_code_1
        random.choice(numbers)   # city_code_2
    ]

def applyNoise(plate):
    background = plate.convert("RGBA")
    noisyTemplates = []
    for noise in noises:
        newPlate = Image.new('RGBA', (600, 132), (0, 0, 0, 0))
        newPlate.paste(background, (0, 0))
        noise_img = Image.open(os.path.join('../Noises/', noise)).convert("RGBA")
        newPlate.paste(noise_img, (0, 0), mask=noise_img)
        noisyTemplates.append(newPlate)
    return noisyTemplates

def applyTransforms(plate):
    transformedTemplates = []
    plate = np.array(plate)
    
    # Rotating clockwise
    for _ in range(3):
        result = imutils.rotate_bound(plate, random.randint(2, 15))
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    # Rotating anticlockwise
    for _ in range(3):
        result = imutils.rotate_bound(plate, random.randint(-15, -2))
        result = Image.fromarray(result)
        transformedTemplates.append(result)
    
    # Scaling up
    for _ in range(3):
        randScale = random.uniform(1.1, 1.3)
        result = cv2.resize(plate, None, fx=randScale, fy=randScale, interpolation=cv2.INTER_CUBIC)
        result = Image.fromarray(result)
        transformedTemplates.append(result)
    
    # Scaling down
    for _ in range(3):
        randScale = random.uniform(0.2, 0.6)
        result = cv2.resize(plate, None, fx=randScale, fy=randScale, interpolation=cv2.INTER_CUBIC)
        result = Image.fromarray(result)
        transformedTemplates.append(result)

    return transformedTemplates


idCounter = 0
total_steps = len(fonts) * len(templates) * permutations * (1 + len(noises) * (1 + (len(transformations) - 1) * 3))
fontsProgBar = tqdm(total=total_steps, desc='Generating Plate...')

for font in fonts:
    if not os.path.exists(font):
        os.mkdir(font)

    letters = []
    with open(f'../Fonts/{font}_namesMap.csv') as nameMapCsv:
        reader = csv.reader(nameMapCsv)
        next(reader)
        letters = [rows[1] for rows in reader]

    for template in templates:
        for i in range(permutations):
            idCounter += 1

            # دریافت ساختار کامل ۸ کاراکتری پلاک
            plate = getNewPlate(template, letters)
            plateName = getPlateName(*plate)

            # خواندن تمام گلیف‌ها (اطمینان از وجود ۸ تصویر گلیف)
            glyphImages = [Image.open(getGlyphAddress(font, g)).convert("RGBA") for g in plate]

            newPlate = Image.new('RGBA', (600, 132), (0, 0, 0, 0))
            background = Image.open(f'../Templates/{template}.png').convert("RGBA")
            newPlate.paste(background, (0, 0))

            # --- چیدمان قطعی و تضمین‌شده برای تمام ۸ کاراکتر ---
            
            # ۱. بخش ۲ رقم اول (سمت چپ)
            x_cursor = 70
            newPlate.paste(glyphImages[0], (x_cursor, 25), mask=glyphImages[0])
            x_cursor += glyphImages[0].size[0] + 5
            
            newPlate.paste(glyphImages[1], (x_cursor, 25), mask=glyphImages[1])
            x_cursor += glyphImages[1].size[0] + 25  # فاصله تا حرف

            # ۲. بخش حرف وسط
            newPlate.paste(glyphImages[2], (x_cursor, 30), mask=glyphImages[2])
            x_cursor += glyphImages[2].size[0] + 25  # فاصله تا ۳ رقم راست

            # ۳. بخش ۳ رقم راست
            for idx in [3, 4, 5]:
                newPlate.paste(glyphImages[idx], (x_cursor, 25), mask=glyphImages[idx])
                x_cursor += glyphImages[idx].size[0] + 5

            # ۴. بخش ۲ رقم کد شهر (مختصات دقیق و ثابت کادر مربع سمت راست)
            # تعیین نقطه ثابت برای شروع کادر کد شهر روی قالب ۶۰۰x۱۳۲
            x_city_start = 450
            newPlate.paste(glyphImages[6], (x_city_start, 25), mask=glyphImages[6])
            
            x_city_second = x_city_start + glyphImages[6].size[0] + 5
            newPlate.paste(glyphImages[7], (x_city_second, 25), mask=glyphImages[7])

            # ذخیره‌سازی پلاک اصلی
            _newPlate = newPlate.resize((312, 70), PIL.Image.LANCZOS)
            _newPlate.save(f"{font}/{plateName}_{template.split('-')[-1]}_{idCounter}.png")
            fontsProgBar.update(1)

            # اعمال نویز
            noisyTemplates = applyNoise(newPlate)
            for noisyTemplate in noisyTemplates:
                idCounter += 1
                _noisyTemplate = noisyTemplate.resize((312, 70), PIL.Image.LANCZOS)
                _noisyTemplate.save(f"{font}/{plateName}_{template.split('-')[-1]}_{idCounter}.png")
                fontsProgBar.update(1)

                # اعمال ترانسفورم‌ها
                transformedTemplates = applyTransforms(noisyTemplate)
                for transformedTemplate in transformedTemplates:
                    idCounter += 1
                    _transformedTemplate = transformedTemplate.resize((312, 70), PIL.Image.LANCZOS)
                    _transformedTemplate.save(f"{font}/{plateName}_{template.split('-')[-1]}_{idCounter}.png")
                    fontsProgBar.update(1)

fontsProgBar.close()