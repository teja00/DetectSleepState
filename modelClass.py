import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_1 = {
            'n_estimators': self.n_estimators,
            'eval_metric': 'map',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': 2/3
        }

        lgbparams = {
            'objective':'binary',
            'boosting_type':'gbdt',
            'learning_rate':0.01,
            'verbosity': 0,
            'random_state': self.random_state
            }
        
       
        models = {
            'xgb_1': xgb.XGBClassifier(**xgb_1),
            'lgb' : lgb.LGBMClassifier(**lgbparams,force_row_wise=True),
            # 'rf': RandomForestClassifier(max_depth=4, min_samples_leaf=100, n_estimators=50, random_state=self.random_state),
        }
        
        return models