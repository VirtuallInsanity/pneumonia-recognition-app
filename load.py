from keras.models import model_from_json
from keras.optimizers import Adam
import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

# saving global session
tf_config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True)

# dynamically grow the memory used on the GPU (without getting errors)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()

def init():
    json_file = open('model_№1.json', 'r')#change name!
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    
    set_session(sess)
    
    loaded_model.load_weights('model_№1_weights.h5')#change name!
    print('Model loaded')

    #compile
    optimizer = Adam(lr=0.0001, decay=1e-6)
    loaded_model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # loss,accuracy = model.evaluate(X_test,y_test)
    # print('loss:', loss)
    # print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model, graph
