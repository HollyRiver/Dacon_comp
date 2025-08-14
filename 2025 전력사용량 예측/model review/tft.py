import pickle
from pytorch_lightning import Trainer

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.examples import get_stallion_data


## data
df_train = pd.read_csv("/root/Dacon_comp/2025 전력사용량 예측/train.csv")
df_test = pd.read_csv("/root/Dacon_comp/2025 전력사용량 예측/test.csv")
building_info = pd.read_csv("/root/Dacon_comp/2025 전력사용량 예측/building_info.csv")

df_train = df_train.rename({pre:new for pre, new in zip(df_train.columns, ["num_date_time", "build_num", "date", "temp", "precip", "wind", "humidity", "sunhour", "sunweight", "power"])}, axis = 1)
df_test = df_test.rename({pre:new for pre, new in zip(df_test.columns, ["num_date_time", "build_num", "date", "temp", "precip", "wind", "humidity", "sunhour", "sunweight"])}, axis = 1)


## information processing
building_info = building_info.replace("-", "0").rename({c:n for c, n in zip(building_info.columns, ["build_num", "build_type", "GFA", "CA", "solar_gen", "ESS", "PCS"])}, axis = 1)\
    .assign(solar_gen = lambda _df : _df.solar_gen.astype("float64"))\
    .assign(ESS = lambda _df : _df.ESS.astype("float64"))\
    .assign(PCS = lambda _df : _df.PCS.astype("float64"))
    
## dttm으로 타입 변경
df_train["date"] = pd.to_datetime(df_train.date)
df_test["date"] = pd.to_datetime(df_test.date)
holilist = ["06-06", "08-15"] ## 2024 추석은 9월 17일
start_date = df_train.date.min()

## train data
del_time = df_train.date - start_date
df_train["time_idx"] = del_time.dt.days*24 + del_time.dt.seconds//3600
df_train["month"] = df_train.date.dt.month.astype(str).astype("category")
df_train["wday"] = df_train.date.dt.weekday.astype(str).astype("category")
df_train["is_holiday"] = df_train.date.astype(str).str[5:10].map(lambda x : 1 if x in holilist else 0)
df_train = df_train.drop(["num_date_time", "date"], axis = 1)
df_train = pd.merge(df_train, building_info, on = "build_num")
df_train["build_num"] = df_train["build_num"].astype(str).astype("category")

## test data
del_time = df_test.date - start_date
df_test["time_idx"] = del_time.dt.days*24 + del_time.dt.seconds//3600
df_test["month"] = df_test.date.dt.month.astype(str).astype("category")
df_test["wday"] = df_test.date.dt.weekday.astype(str).astype("category")
df_test["is_holiday"] = 0 ## 8.25 이후 휴일 없음
df_test = df_test.drop(["num_date_time", "date"], axis = 1)
df_test = pd.merge(df_test, building_info, on = "build_num")
df_test["build_num"] = df_test["build_num"].astype(str).astype("category")

## Trainer
max_prediction_length = 120     ## validation length
max_encoder_length = 336        ## lookback (14일)
training_cutoff = 1919          ## 0 to 1919 (1920 periods)

training = TimeSeriesDataSet(
    df_train[df_train.time_idx <= training_cutoff],
    time_idx = "time_idx",
    target = "power",
    group_ids = ["build_num"],
    min_encoder_length = max_encoder_length // 2,
    max_encoder_length = max_encoder_length,
    min_prediction_length = 1,
    max_prediction_length = max_prediction_length,
    static_categoricals = ["build_num", "build_type"],
    static_reals = ["GFA", "CA", "solar_gen", "ESS", "PCS"],
    # variable_groups = {}, ## 아직은 없는듯?
    time_varying_known_categoricals = ["wday", "month"],
    time_varying_known_reals = ["temp", "precip", "wind", "humidity", "is_holiday"],
    time_varying_unknown_reals = ["sunhour", "sunweight"],
    target_normalizer = GroupNormalizer(groups = ["build_num"], transformation = "softplus"),
    add_relative_time_idx = True,
    add_target_scales = True, ## 정규화되지 않은 시계열의 중심과 스케일을 피쳐로
    add_encoder_length = True ## 인코더의 길이를 static feature에 추가
)

## create validation set
validation = TimeSeriesDataSet.from_dataset(
    training, df_train, predict = True, stop_randomization = True
)

## create dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(
    train = True, batch_size = batch_size, num_workers = 8
)
val_dataloader = validation.to_dataloader(
    train = False, batch_size = batch_size, num_workers = 0
)


## Learning
pl.seed_everything(42)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate = 0.03,
    hidden_size = 128,
    attention_head_size = 4,
    dropout = 0.2,
    hidden_continuous_size = 64,
    loss = QuantileLoss(),
    log_interval = 10,
    # reduce_on_plateau_patience = 1000
)


## Trainer setting
early_stop_callback = EarlyStopping(
    monitor = "val_loss", min_delta = 1e-4, patience = 10, verbose = False, mode = "min"
)
lr_logger = LearningRateMonitor() ## log the learning rate
logger = TensorBoardLogger("lightning_logs") ## logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs = 150, ## 고작 50?
    accelerator = "gpu",
    enable_model_summary = True,
    gradient_clip_val = 1.0,
    limit_train_batches = 30, ## comment in for training, running validation every 30 batches
    callbacks = [lr_logger, early_stop_callback],
    logger = logger
)


trainer.fit(tft, train_dataloader, val_dataloader)

torch.save(trainer, "TFT_trainer.trch")
torch.save(trainer.model, "TFT_model.trch")