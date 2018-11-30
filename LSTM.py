import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Embedding, Input, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K

def clipped_mse(y_true, y_pred):
    return K.mean(K.square(K.clip(y_pred, 0., 20) - K.clip(y_true, 0., 20)), axis=-1)

model_filename = 'LSTM-best.h5'

callbacks = [ReduceLROnPlateau(monitor='loss', factor=.9, patience=3, verbose=1, epsilon=0.0001),
             ModelCheckpoint(model_filename, monitor='loss', save_best_only=True, verbose=1),
             EarlyStopping(monitor='loss', patience=5, verbose=1)]

if __name__ == "__main__":

    items_df = pd.read_csv('Data/items.csv')
    shops_df = pd.read_csv('Data/shops.csv')
    icats_df = pd.read_csv("Data/item_categories.csv")
    train_df = pd.read_csv("Data/sales_train.csv.gz")
    test_df = pd.read_csv('Data/test.csv.gz')  # 214200 rows

    test_shops = test_df.shop_id.unique()
    train_df = train_df[train_df.shop_id.isin(test_shops)]
    test_items = test_df.item_id.unique()
    train_df = train_df[train_df.item_id.isin(test_items)]

    print('train:', train_df.shape, 'test:', test_df.shape, 'items:', items_df.shape, 'shops:', shops_df.shape)

    shops_df['city_id'] = shops_df.shop_name.apply(lambda x: str.replace(x, '!', '')).apply(lambda x: x.split(' ')[0])
    shops_df['city_id'] = pd.Categorical(shops_df['city_id']).codes

    icats_df['item_category_group'] = icats_df['item_category_name'].apply(lambda x: str(x).split(' ')[0])
    icats_df['item_category_group'] = pd.Categorical(icats_df['item_category_group']).codes

    train_piv = train_df.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day',
                                     aggfunc='sum').fillna(0.0)
    train_piv = train_piv.reset_index()

    train_piv = pd.merge(train_piv, shops_df, on=['shop_id'], how='left')

    """
    Input Data
    """
    data = train_piv.drop([33], axis=1)
    col = np.arange(33)
    X_train = np.array([[x] for x in data[col].values])
    Y_train = train_piv[33].values

    shop_id = train_piv['shop_id'].values
    item_id = train_piv['item_id'].values
    city_id = train_piv['city_id'].values

    """
    ****************************************************
    """

    n_items = train_piv.item_id.max()  # train_piv.item_id.nunique()
    n_shops = train_piv.shop_id.max()  # train_piv.shop_id.nunique()
    n_citys = train_piv.city_id.max()  # train_piv.shop_id.nunique()

    item_input = Input(shape=(1,), name='Items')
    item_embed = Embedding(n_items + 1, 32, input_length=1, name='ItemEmbedding')(item_input)
    item_vec = Flatten(name='ItemsFlatten')(item_embed)
    item_vec = Dense(16, name='Item_Dense_1')(item_vec)

    shop_input = Input(shape=(1,), name='Shops')
    shop_embed = Embedding(n_shops + 1, 8, input_length=1, name='ShopEmbedding')(shop_input)
    shop_vec = Flatten(name='ShopsFlatten')(shop_embed)
    shop_vec = Dense(16, activation='relu', name='Shop_Dense_1')(shop_vec)

    city_input = Input(shape=(1,), name='City')
    city_embed = Embedding(n_shops + 1, 8, input_length=1, name='CityEmbedding')(city_input)
    city_vec = Flatten(name='CitysFlatten')(city_embed)
    city_vec = Dense(16, activation='relu', name='City_Dense_1')(city_vec)

    # LSTM
    sale_input = Input(shape=(1, 33), name='Sales')
    sale_lstm = LSTM(64, input_shape=(1, 33))(sale_input)
    sale_lstm = Dense(16, activation='relu', name='Sale_LSTM')(sale_lstm)

    concat = keras.layers.concatenate([item_vec, shop_vec, city_vec, sale_lstm], name='Concat')

    fc_1 = Dense(32, activation='relu', name='fc_1')(concat)
    fc_2 = BatchNormalization()(fc_1)
    fc_3 = Dense(8, activation='relu', name='fc_2')(fc_2)
    fc_4 = BatchNormalization()(fc_3)
    output = Dense(1, activation='relu', name='Output')(fc_4)

    optimizer = Adam()

    model = keras.Model([item_input, shop_input, city_input, sale_input], output)

    model.compile(optimizer=optimizer, loss=clipped_mse)

    model.fit([item_id, shop_id, city_id, X_train], Y_train, validation_split=0.1, epochs=10, batch_size=256)