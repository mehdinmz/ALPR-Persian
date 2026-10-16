from hezar.models import Model
from hezar.preprocessors import preprocessor
model = Model.load("D:\Projects\ALPR-Persian\ocr\Models\crnn-fa-license-plate-recognition-v2",
                   load_locally= True,
                   )
preprocessor = preprocessor.Preprocessor.load(subfolder="preprocessor", hub_or_local_path="D:\Projects\ALPR-Persian\ocr\Models\crnn-fa-license-plate-recognition-v2")
plate_text = model.predict(preprocess=preprocessor, device="cpu",inputs="D:\Projects\ALPR-Persian\detection\plate_0.jpg")
print(plate_text[0].text)  # Persian text of mixed numbers and characters might not show correctly in the console

