import pandas as pd
import networkx as nx

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

from trenchant_utils import make_hin
from trenchant_utils import inner_connections

df = pd.read_parquet('/media/pauloricardo/basement/commodities_usecase/soybean_corn_4w1h.parquet')
path = '/media/pauloricardo/basement/commodities_usecase/'
fine_tune = 'fine-tuned-twelve-months-soy'
fine_tune_dict = {
  'fine-tuned-twelve-weeks-corn': {'interval_feature': 'WeekYear', 'label': 'WeekYearCornTrend', 'commodity': 'corn', 'interval': 'week',},
  'fine-tuned-twenty_four-weeks-corn': {'interval_feature': 'WeekYear', 'label': 'WeekYearCornTrend', 'commodity': 'corn', 'interval': 'week',},
  'fine-tuned-fourty_eight-weeks-corn': {'interval_feature': 'WeekYear', 'label': 'WeekYearCornTrend', 'commodity': 'corn', 'interval': 'week',},
  'fine-tuned-twelve-weeks-soy': {'interval_feature': 'WeekYear', 'label': 'WeekYearSoyTrend', 'commodity': 'soybean', 'interval': 'week',},
  'fine-tuned-twenty_four-weeks-soy': {'interval_feature': 'WeekYear', 'label': 'WeekYearSoyTrend', 'commodity': 'soybean', 'interval': 'week',},
  'fine-tuned-fourty_eight-weeks-soy': {'interval_feature': 'WeekYear', 'label': 'WeekYearSoyTrend', 'commodity': 'soybean', 'interval': 'week',},
  'fine-tuned-three-months-corn': {'interval_feature': 'MonthYear', 'label': 'MonthYearCornTrend', 'commodity': 'corn', 'interval': 'month',},
  'fine-tuned-six-months-corn': {'interval_feature': 'MonthYear', 'label': 'MonthYearCornTrend', 'commodity': 'corn', 'interval': 'month',},
  'fine-tuned-twelve-months-corn': {'interval_feature': 'MonthYear', 'label': 'MonthYearCornTrend', 'commodity': 'corn', 'interval': 'month',},
  'fine-tuned-three-months-soy': {'interval_feature': 'MonthYear', 'label': 'MonthYearSoyTrend', 'commodity': 'soybean', 'interval': 'month',},
  'fine-tuned-six-months-soy': {'interval_feature': 'MonthYear', 'label': 'MonthYearSoyTrend', 'commodity': 'soybean', 'interval': 'month',},
  'fine-tuned-twelve-months-soy': {'interval_feature': 'MonthYear', 'label': 'MonthYearSoyTrend', 'commodity': 'soybean', 'interval': 'month',},
}


# load model
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model = SentenceTransformer(f'{path}fine-tuned-models/{fine_tune}')

df[fine_tune] = list(model.encode(df['Headlines'].to_list()))

# make network according to fine_tune
G = make_hin(df, embedding_feature=fine_tune, date_feature=fine_tune_dict[fine_tune]['interval_feature'], commodities_feature=fine_tune_dict[fine_tune]['label'])
G = inner_connections(G)
nx.write_gpickle(G, f"/media/pauloricardo/basement/commodities_usecase/{fine_tune_dict[fine_tune]['commodity']}_{fine_tune_dict[fine_tune]['interval']}_{fine_tune}.gpickle")