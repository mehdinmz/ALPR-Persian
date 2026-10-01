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


def getPlateName(n1, n2, l, n3, n4, n5, c1, c2):
    return f'{n1}{n2}{l}{n3}{n4}{n5}_{c1}{c2}'

def getGlyphAddress(font, glyphName):
    t = "..\\"
    return f'{t}Glyphs/{font}/{glyphName}_trim.png'

def getNewPlate(template_name):
    available_letters = {
        "shakhsi": ["sin", "sad", "beh", "jim", "dal", "ta", "ghaf", "lam","mim","noon","vav","he", "ye"],
        "malolin": ["zhe"],
        "sepah":["seh"],
        "police":["peh"],
        "defa" : ["zeh"],
        "artesh": ["shin"],
        "Niromosalah": ["feh"],
        "gozar": ["gaf"],
        "taxi": ["teh"],
        "ommomi": ["ein"],
        "dolati" : ["alef"],
        "diplomat" : ["D"],
        "service" : ["S"],
        "keshavarzi" : ["kaf"]

    }
    template_name = template_name.split('-')[-1].split('.')[0]
    available_letters = available_letters.get(template_name, [])
    # ب ج د س
    # ص ط ق ل
    # م ن و ه‍ ی
    """
    تولید ساختار کامل پلاک (۸ کاراکتر):
    [n1, n2, letter, n3, n4, n5, city1, city2]
    """
    valid_letters = available_letters.copy()
    selected_letter = random.choice(valid_letters)
    # print(f"Selected letter for template '{template_name}': {selected_letter}")
    #full = is_shakhsi
    city_code_1 = random.choice(range(1, 10))
    if template_name in ['sepah', 'police', 'defa', 'artesh', 'Niromosalah', 'diplomat', 'service']:
        city_code_2 = random.choice((0, city_code_1))   # Ensure city_code_2 is not zero for these templates
    else:
        city_code_2 = random.choice(range(0, 10))
    ret = [
        random.choice(range(1, 10)),  # n1
        random.choice(range(1, 10)),  # n2
        selected_letter,         # letter
        random.choice(range(1, 10)),  # n3
        random.choice(range(1, 10)),  # n4
        random.choice(range(1, 10)),  # n5
        city_code_1,  # city_code_1
        city_code_2   # city_code_2
        #0
    ]
    print("plate:",ret)
    return ret

def applyNoise(plate):
    # Noises
    n = "..\\Noises"
    noises = os.listdir(n)
    transformations = ['rotate_right', 'rotate_left', 'zoom_in', 'zoom_out', 'prespective_transform']

    # Count of permutations
    permutations = 1

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

def load_glyphs(font,plate):
    res = []
    for g in plate:
         if os.path.exists(getGlyphAddress(font, g)):
            res.append(Image.open(getGlyphAddress(font, g)).convert("RGBA"))
         else:
            res.append(Image.new("RGBA", (50, 100), (0, 0, 0, 0)))  # Create a transparent image if glyph doesn't exist
    return res


total_steps = 100 #len(fonts) * len(templates) * permutations * (1 + len(noises) * (1 + (len(transformations) - 1) * 3))
fontsProgBar = tqdm(total=total_steps, desc='Generating Plate...')

# for font in fonts:
#     if not os.path.exists(font):
#         os.mkdir(font)
def main():
    # Characters of Letters and Numbers in Plates
    numbers = [str(i) for i in range(0, 10)]

    # Fonts and Templates
    font = 'roya_bold'
    t = "..\\Templates\\"
    #defining cursor for each template type
    cursor = {
        "dolati": {"x":65,"y":30,"s_2r":6, "s_3r":7, "s_l":32},
        "gozar": {"x":65,"y":30,"s_2r":5, "s_3r":5, "s_l":30},
        "nezami": {"x":73,"y":30,"s_2r":5, "s_3r":5, "s_l":27},
        "malolin va janbazan": {"x":70,"y":32,"s_2r":6, "s_3r":7, "s_l":24},
        "shakhsi": {"x":73,"y":30,"s_2r":5, "s_3r":6, "s_l":23},
        "siasi": {"x":70,"y":30,"s_2r":5, "s_3r":8, "s_l":22},
        "tranzit": {"x":50,"y":5,"s_2r":5, "s_3r":7, "s_l":19},
        "tashrifat": {"x":332,"y":30,"s_2r":4, "s_3r":4, "s_l":-29},
        "ommomi": {"x":65,"y":30,"s_2r":6, "s_3r":7, "s_l":30},
    }
    whites = [
                'template-dolati.png',
                'template-Niromosalah.png',
                'template-defa.png',
                'template-tashrifat.png',
                'template-sepah.png',
                'template-police.png'
            ]
    
    # letters = []
    # with open(f'../Fonts/{font}_namesMap.csv') as nameMapCsv:
    #     reader = csv.reader(nameMapCsv)
    #     next(reader)
    #     letters = [rows[1] for rows in reader]

    for templates in os.listdir(t):

        base = templates in ['shakhsi', 'malolin va janbazan', 'tranzit', 'ommomi', 'dolati','nezami','siasi']

        for template in os.listdir(t + templates):
            idCounter = 0
            white = template in whites

            # دریافت ساختار کامل ۸ کاراکتری پلاک
            plate = getNewPlate(template)
            plateName = getPlateName(*plate)

            # خواندن تمام گلیف‌ها
            # اطمینان از وجود ۸ تصویر گلیف
            glyphImages = load_glyphs(font,plate)
            if glyphImages[2] is not None:
            # اگر پلاک جزو پلاک‌های سفید است，
            # رنگ گلیف‌ها را از سیاه به سفید تغییر بده
                if white:

                    whiteGlyphImages = []

                    for glyph in glyphImages:
                        # جدا کردن کانال‌های RGBA
                        r, g, b, a = glyph.split()

                        # تبدیل رنگ کاراکتر از سیاه به سفید
                        # کانال Alpha بدون تغییر باقی می‌ماند
                        whiteGlyph = Image.new(
                            "RGBA",
                            glyph.size,
                            (255, 255, 255, 0)
                        )

                        whiteGlyph.putalpha(a)

                        whiteGlyphImages.append(whiteGlyph)

                    glyphImages = whiteGlyphImages

                print(templates, template,plateName)

                newPlate = Image.new(
                    'RGBA',
                    (600, 132),
                    (0, 0, 0, 0)
                )

                background = Image.open(
                    f'{t}{templates}\\{template}'
                ).convert("RGBA")

                newPlate.paste(background, (0, 0))

                x_cursor = cursor[templates]["x"]
                y_cursor = cursor[templates]["y"]

                s_2r = cursor[templates]["s_2r"]
                s_3r = cursor[templates]["s_3r"]
                s_l = cursor[templates]["s_l"]

                newPlate.paste(
                    glyphImages[0],
                    (x_cursor, y_cursor),
                    mask=glyphImages[0]
                )

                x_cursor += 50 + s_2r

                # print(x_cursor)

                newPlate.paste(
                    glyphImages[1],
                    (x_cursor, y_cursor),
                    mask=glyphImages[1]
                )

                x_cursor += 45 + s_l  # فاصله تا حرف

                # print(x_cursor)

                # ۲. بخش حرف وسط
                if base and templates != 'malolin va janbazan':

                    newPlate.paste(
                        glyphImages[2],
                        (x_cursor, y_cursor + 3),
                        mask=glyphImages[2]
                    )

                x_cursor += 65 + s_l  # فاصله تا ۳ رقم راست

                # print("size", glyphImages[2].size[0])

                for idx in [3, 4, 5]:

                    newPlate.paste(
                        glyphImages[idx],
                        (x_cursor, y_cursor),
                        mask=glyphImages[idx]
                    )

                    x_cursor += 50 + s_3r

                    # print(x_cursor)

                # ۴. بخش ۲ رقم کد شهر
                # مختصات دقیق و ثابت کادر مربع سمت راست

                if base:

                    x_city_start = 480

                    newPlate.paste(
                        glyphImages[6],
                        (x_city_start, y_cursor + 5),
                        mask=glyphImages[6]
                    )

                    # print(plate[-1])

                    if plate[-1] == 0:

                        # print(y_cursor)
                        y_cursor -= -23

                    x_city_second = x_city_start + 50 + s_2r

                    newPlate.paste(
                        glyphImages[7],
                        (x_city_second, y_cursor + 5),
                        mask=glyphImages[7]
                    )

                    # print(x_city_second)

                # Image.Image.show(newPlate)
                # ذخیره‌سازی پلاک اصلی
                if not os.path.exists(f"{font}/{plateName}"):
                                        os.mkdir(f"{font}/{plateName}")
                _newPlate = newPlate.resize((312, 70), PIL.Image.LANCZOS)
                _newPlate.save(f"{font}/{plateName}/{idCounter}.png")
                fontsProgBar.update(1)

                # اعمال نویز
                noisyTemplates = applyNoise(newPlate)
                for noisyTemplate in noisyTemplates:
                    idCounter += 1
                    _noisyTemplate = noisyTemplate.resize((312, 70), PIL.Image.LANCZOS)
                    _noisyTemplate.save(f"{font}/{plateName}/{idCounter}.png")
                    fontsProgBar.update(1)

                    # اعمال ترانسفورم‌ها
                    transformedTemplates = applyTransforms(noisyTemplate)
                    for transformedTemplate in transformedTemplates:
                        idCounter += 1
                        _transformedTemplate = transformedTemplate.resize((312, 70), PIL.Image.LANCZOS)
                        _transformedTemplate.save(f"{font}/{plateName}/{idCounter}.png")
                        fontsProgBar.update(1)
main()
fontsProgBar.close()