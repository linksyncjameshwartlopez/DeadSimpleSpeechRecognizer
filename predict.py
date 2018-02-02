from preprocess import *
from helpers import *


loaded_model = load_model_from_disk('cnn_model')
# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
print(predict('cat_2.wav', model=loaded_model))