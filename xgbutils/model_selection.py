import copy
import hyperopt
import xgboost as xgb


class ParamSearchCV(object):

    def __init__(self, param_space, num_evals, cv_metric,
                 fit_params=None, cv_params=None):
        """
        Parameters
        ----------
        param_space: {param_name: hyperopt_space}
        num_evals: int
            Number of hyperopt evaluations.
        cv_metric: string
            Name of cv_metric to minimize.
        fit_params: dict()
        cv_params: dict()
        """
        self.param_space = param_space
        self.num_evals = num_evals
        self.__cv_metric = cv_metric
        self.fit_params = copy.copy(fit_params) or dict(params=dict())
        self.cv_params = copy.copy(cv_params) or dict(params=dict())

        self.best_params_ = dict()
        self.best_estimator_ = None
        self.num_boost_round_ = None

    @property
    def _cv_metric(self):
        return "test-%s-mean" % self.__cv_metric

    @staticmethod
    def _combine_params(hp_params, input_params):
        for param_name, param_val in hp_params.iteritems():
            input_params["params"][param_name] = param_val
        return input_params

    def _cv_objective(self, dtrain):

        def hyperopt_objective(kwargs):
            if "max_depth" in kwargs:
                kwargs["max_depth"] = int(kwargs["max_depth"])

            params = self._combine_params(hp_params=kwargs, input_params=self.cv_params)
            bst = xgb.cv(dtrain=dtrain, **params)
            loss = bst[self._cv_metric].min()
            return loss

        return hyperopt_objective

    def fit(self, dtrain):
        f = self._cv_objective(dtrain=dtrain)
        best_params_ = hyperopt.fmin(f, space=self.param_space, algo=hyperopt.tpe.suggest,
                                     max_evals=self.num_evals)
        best_params_["max_depth"] = int(best_params_["max_depth"])
        self.best_params_ = best_params_

        # choose number of trees
        cv_params = self._combine_params(self.best_params_, self.cv_params)
        bst = xgb.cv(dtrain=dtrain, **cv_params)
        self.num_boost_round_ = bst[self._cv_metric].argmin()
        self.fit_params["num_boost_round"] = self.num_boost_round_

        # re-fit using number of trees found from cross validation
        train_params = self._combine_params(hp_params=self.best_params_, input_params=self.fit_params)
        self.best_estimator_ = xgb.train(dtrain=dtrain, **train_params)

    def predict(self, dtest):
        return self.best_estimator_.predict(dtest)
