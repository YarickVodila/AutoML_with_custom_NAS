import tensorflow as tf # ==2.10
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Normalization
from tensorflow.keras.optimizers import Adam
import optuna
from optuna.samplers import TPESampler
from IPython.display import clear_output


class TabularDataClassifier:
    features_train = None
    target_train = None
    features_val = None
    target_val = None
    verbose = None
    batch_size = None 
    epochs = None
    history = None
    model = None

    def __init__(self, n_trails, n_class = 2):
        '''
        Метод инициализации модели

        :param n_trails: количество итераций подбора архитектуры (int);
        :param n_class: Если задача бинарной классификации, то параметру следует назначить значение 2,
         иначе значение равное количеству классов (int);
        
        '''
         
        self.n_trails = n_trails
        self.n_class = n_class

    def __create_model(self, input_shape, layers, units, dropouts):
        '''
        Закрытый метод, который создаёт модель по параметрам.

        :param input_shape: размер входных данных;
        :param layers: количество скрытых слоёв;
        :param units: количество нейронов;
        :param dropouts: параметр rate у слоя Dropout;
        
        :return: Возвращаем архитектуру модели
        '''
    
        model = Sequential()
        for i in range(layers):
            if i==0:
                model.add(Dense(units, input_shape = (input_shape,), activation = 'relu'))
                model.add(Dropout(dropouts))
                model.add(Normalization()) # выполняет пофункциональную нормализацию входных функций.
            else:
                model.add(Dense(units//(i*2), activation = 'relu'))
                model.add(Dropout(dropouts))
                model.add(Normalization())
        

        if self.n_class == 2:
            model.add(Dense(1, activation = 'sigmoid'))
            optimizer = Adam(learning_rate = 0.001)
            model.compile(loss ='binary_crossentropy', optimizer = optimizer, metrics = ['BinaryAccuracy', 'AUC'])
        else:

            model.add(Dense(self.n_class, activation = 'softmax'))
            optimizer = Adam(learning_rate = 0.001)
            model.compile(loss ='sparse_categorical_accuracy', optimizer = optimizer, metrics = ['accuracy'])

        model.summary()

        return model

    def __nas(self, trail):
        layers = trail.suggest_int('count_layers', 1, 4, log = True)
        units = trail.suggest_categorical("units", [32, 64, 128, 256, 512])
        dropouts = trail.suggest_categorical("dropout", [0.1, 0.2, 0.25, 0.3])

        model = self.__create_model(self.features_train.shape[1], layers, units, dropouts)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=20, mode='max',
                                                     min_delta=0.0001, restore_best_weights = True)
        
        model.fit(self.features_train, self.target_train, verbose = self.verbose, batch_size = self.batch_size, epochs = self.epochs, 
            validation_data=(self.features_val, self.target_val), callbacks=[callback])
        
        bin_acc = model.evaluate(self.features_val, self.target_val)[1]
        clear_output(wait=True)
        
        return bin_acc

    def fit(self, X, y, verbose, batch_size, epochs, validation_data = None):
        self.features_train = X
        self.target_train = y

        if validation_data:
            self.features_val = validation_data[0]
            self.target_val = validation_data[1]  

        self.verbose = verbose
        self.batch_size = batch_size 
        self.epochs = epochs
        
        sampler = TPESampler(seed=12345)
        study = optuna.create_study(study_name="ANN", direction="maximize", sampler=sampler)
        study.optimize(self.__nas, n_trials = self.n_trails)

        clear_output(wait=True)
        trial = study.best_trial
        

        model = self.__create_model(self.features_train.shape[1], trial.params.get('count_layers'), trial.params.get('units'),
                                   trial.params.get('dropout'))
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=20, mode='max',
                                                     min_delta=0.0001, restore_best_weights = True)
        
        history = model.fit(self.features_train, self.target_train, verbose = self.verbose, batch_size = self.batch_size,
                             epochs = self.epochs, validation_data = (self.features_val, self.target_val), callbacks = [callback])
        
        print('\n','#' * 40, 'END', '#' * 41, end='\n\n')
        print(f"Best metrics: {trial.value}", end='\n\n')
        print('#' * 34, 'Best model was training', '#' * 34, end='\n\n')
        
        self.model = model
        self.history = history