import paddleclas
model = paddleclas.PaddleClas(model_name="person_attribute")
result = model.predict(input_data="pulc_demo_imgs/person_attribute/090004.jpg")
print(next(result))