import pytesseract
from pytesseract import Output
import cv2
from pdf2image import convert_from_path
from dateutil.parser import parse

# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

pdfs = r"treds_invoices_for_cxsphere\201704211557354971.uni-538.pdf"
pages = convert_from_path(pdfs, 350)

i = 1
for page in pages:
    image_name = "Page_" + str(i) + ".png"  
    page.save(image_name, "JPEG")
    i = i+1        

img = cv2.imread(image_name)

keywords = [
    "INVOICE",
    "INVOICE#",
    "INVOICE #",
    "Invoice no.",
    "INVOICE DATE",
    "Total",
    # "TAX",
    "AMOUNT",
    "AMOUNTS",
    "Date",
    "Due date:",
    "Bill To",
    "Bill",
    "Ship To",
    "Ship",
    "From",
    "TO",
] 

def findTotal(nearest_words):
    for chr in nearest_words:
        if not any(c.isalpha() for c in chr):
            return chr
    return "Not found"

def findDate(nearest_words, fuzzy=False):
    for chr in nearest_words:
        try:
            parse(chr, fuzzy=fuzzy)
            return chr
        except ValueError:
            continue
    return "Not found"

def distance(coord1, coord2, axis, x_stretch=1, y_stretch=1):
    """Retruns the distance-squared of two points (x1,y1), (x2,y2)
    large x_/ y_stretch => shorter dists for point in same horz/vert line"""
    sign = lambda a: (a > 0) - (a < 0)
    x1, y1 = coord1
    x2, y2 = coord2
    if axis in ["x", "X"]:
        x_stretch, y_stretch = 1, 100
        sign_d = sign(x1 - x2)  ## k should be right of key
    elif axis in ["y", "Y"]:
        x_stretch, y_stretch = 6, 1
        sign_d = sign(y2 - y1)  ## k should be below key
    else:
        raise Exception("'axis' must be either 'x' or 'y'")

    d = ((x2 - x1) * x_stretch) ** 2 + ((y2 - y1) * y_stretch) ** 2
    return d

def nearest_text(dic, key, axis):
    """ locate the nearest text along x-/ y-axis """
    ## TODO: Fetch all the nearest words of all the occurances of the given keyword
    textList = []
    for i in dic:
        ## TODO: Check the match of the key with each word in loop greater than equal to theshold
        if i[0].upper() == key:
            all_distances = {
                k[0]: distance([i[1], i[2]], [k[1], k[2]], axis)
                for k in set(dic) - {key}
            }
            text = sorted(all_distances, key=all_distances.get)
            textList.append(text[1])
    
    return textList

dic = pytesseract.image_to_data(img, output_type=Output.DICT)

processed_dic = [
    (dic["text"][i], dic["left"][i], dic["top"][i])
    for i in range(len(dic["text"]))
    if dic["text"][i] not in ["", " "]
]

print(processed_dic)

for word in keywords:
    horizontal_word = nearest_text(processed_dic, word.upper(), axis="X")
    vertical_word = nearest_text(processed_dic, word.upper(), axis="Y")
    if len(horizontal_word) > 0 or len(vertical_word) > 0:
        if word.upper() == 'TOTAL' or word.upper() == 'AMOUNT':
            print ('%20s'%word +' --> ', '%20s'%(findTotal(horizontal_word + vertical_word)) + '\n')
        elif word.upper() == 'DATE':
            print ('%20s'%word +' --> ', '%20s'%(findDate(horizontal_word + vertical_word)) + '\n')
        else:
            print ('%20s'%word +' --> ', '%20s'%horizontal_word + ' --> ', '%20s'%vertical_word + '\n')