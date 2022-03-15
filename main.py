from data import load_data
from models import DNN_model, CNN_model
from tensorflow.keras.callbacks import EarlyStopping

# retrieve data
X_train, X_test, y_train, y_test = load_data(model_type= "DNN")

# load the model
model = DNN_model()
# set a callback
callback = EarlyStopping(patience = 3)
# train the model
h = model.fit(X_train, y_train,
              validation_split = 0.2,
              epochs = 10, batch_size= 32,
              callbacks = [callback])
# evaluate the model
model.evaluate(X_test, y_test)
# save the model
model.save('Models/model.h5')
