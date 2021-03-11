import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression

class Evaluator:
    regularization_parameters = [0.1*alpha for alpha in range(1, 31)]

    def evaluate(self, n_samples, n_features):
        beta = np.random.rand(n_features+1)
        train_X, test_X = (np.random.standard_normal(size=(n_samples, n_features)) for _ in range(2))
        train_Y, test_Y  = (self.simulate_Y(X, beta) for X in (train_X, test_X))
        simple_model = self.simple_model().fit(train_X, train_Y)
        simple_score = simple_model.score(test_X, test_Y)
        scores = [self.evaluate_reg(reg_param, train_X, test_X, train_Y, test_Y)
                  for reg_param in self.regularization_parameters]
        return simple_score, np.array(scores)

    def evaluate_reg(self, reg_param, train_X, test_X, train_Y, test_Y):
        model = self.regularized_model(reg_param).fit(train_X, train_Y)    
        return model.score(test_X, test_Y)

    def main_plot(self, n_sim=10):
        simple_sum, score_sum = (0, 0)
        for _ in range(n_sim):
            simple_score, scores = self.evaluate(100, 1000)
            simple_sum += simple_score
            score_sum += scores
        score_sum /= 10
        simple_sum /= 10
        plt.plot(e.regularization_parameters, score_sum, ".", color="red", label="Regularized");
        plt.axhline(simple_sum, label="Unregularized")
        plt.xlabel("Regularization parameter")
        plt.ylabel("Mean model score")
        plt.legend()
        plt.title(self.name)


class LinearEvaluator(Evaluator):
    simple_model=LinearRegression
    regularized_model=Lasso
    name="LinearRegression"

    sigma=0.1

    def simulate_Y(self, X, beta):
        EY = beta[0]+np.sum(beta[1:]*X, axis=1)
        Y = EY + self.sigma*np.random.standard_normal(EY.shape)
        return Y
        

class LogisticEvaluator(Evaluator):
    simple_model = lambda _: LogisticRegression(penalty="none")
    regularized_model = lambda _, alpha: LogisticRegression(penalty="l1", C=alpha, solver="liblinear")
    name="LogisticRegression"

    def simulate_Y(self, X, beta):
        eta = beta[0]+np.sum(beta[1:]*X, axis=1)
        p = np.exp(eta)/(1+np.exp(eta))
        return np.random.rand(p.size)<p


LinearEvaluator().main_plot()
plt.show()
LogisticEvaluator().main_plot()
plt.show()
