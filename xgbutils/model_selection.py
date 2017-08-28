import copy
import hyperopt
import xgboost as xgb


class ParamOptimizeCV(object):

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
            Parameters we'll pass to to xgb.train
        cv_params: dict()
            Parameters we'll pass to xgb.cv
        """
        self.param_space = param_space
        self.num_evals = num_evals
        self.__cv_metric = cv_metric
        self.fit_params = copy.copy(fit_params) or dict(params=dict())
        self.cv_params = copy.copy(cv_params) or dict(params=dict())

        self.optimized_params_ = dict()
        self.best_estimator_ = None
        self.num_boost_round_ = None

    @property
    def _cv_metric(self):
        return "test-%s-mean" % self.__cv_metric

    @staticmethod
    def _update_params(hp_params, input_params):
        """
        Parameters
        ----------
        hp_params: {param_name: param_value}
            Trainable parameters.
        input_params: {param_name: param_value}
            Non-trainable parameters.
        """
        for param_name, param_val in hp_params.iteritems():
            input_params["params"][param_name] = param_val
        return input_params

    def _cv_objective(self, dtrain):
        """
        Given dtrain, return function of trainable parameters that computes cross validated loss.

        Returns
        -------
        kwargs -> float
        """

        def hyperopt_objective(kwargs):
            if "max_depth" in kwargs:
                kwargs["max_depth"] = int(kwargs["max_depth"])

            params = self._update_params(hp_params=kwargs, input_params=self.cv_params)
            bst = xgb.cv(dtrain=dtrain, **params)
            loss = bst[self._cv_metric].min()
            return loss

        return hyperopt_objective

    def fit(self, dtrain):
        """
        Choose optimal parameters and number of trees using cross-validation. Then re-fit model on entire dataset.

        Parameters
        ----------
        dtrain: xgb.DMatrix
        """
        if "evals" not in self.fit_params:
            self.fit_params = dict(self.fit_params.items() + {"evals": [(dtrain, "train")]}.items())

        f = self._cv_objective(dtrain=dtrain)
        optimized_params_ = hyperopt.fmin(f, space=self.param_space, algo=hyperopt.tpe.suggest,
                                          max_evals=self.num_evals)
        optimized_params_["max_depth"] = int(optimized_params_["max_depth"])
        self.optimized_params_ = optimized_params_

        # choose number of trees
        cv_params = self._update_params(self.optimized_params_, self.cv_params)
        bst = xgb.cv(dtrain=dtrain, **cv_params)
        self.num_boost_round_ = bst[self._cv_metric].argmin()
        self.fit_params["num_boost_round"] = self.num_boost_round_

        # re-fit using number of trees found from cross validation
        train_params = self._update_params(hp_params=self.optimized_params_, input_params=self.fit_params)
        self.best_estimator_ = xgb.train(dtrain=dtrain, **train_params)

    def predict(self, dpredict, **kwargs):
        """
        Parameters
        ----------
        dpredict: xgb.DMatrix

        Returns
        -------
        array like
        """
        return self.best_estimator_.predict(data=dpredict, **kwargs)
