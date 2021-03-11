import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression

class Evaluator:
    simple_model=LinearRegression
    regularized_model=Lasso
    sigma=0.1
    regularization_parameters = [0.1*alpha for alpha in range(1, 31)]

    def simulate_Y(self, X, beta):
        EY = beta[0]+np.sum(beta[1:]*X, axis=1)
        Y = EY + self.sigma*np.random.standard_normal(EY.shape)
        return Y
        
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

class LogisticEvaluator(Evaluator):
    simple_model = lambda _: LogisticRegression(penalty="none")
    regularized_model = lambda _, alpha: LogisticRegression(penalty="l1", C=alpha, solver="liblinear")

    def simulate_Y(self, X, beta):
        eta = beta[0]+np.sum(beta[1:]*X, axis=1)
        p = np.exp(eta)/(1+np.exp(eta))
        return np.random.rand(p.size)<p


def simulate_Y(X, beta, sigma):
    EY = beta[0]+np.sum(beta[1:]*X, axis=1)
    Y = EY + sigma*np.random.standard_normal(EY.shape)
    return Y

def simulate_bernoulli(X, beta):
    eta = beta[0]+np.sum(beta[1:]*X, axis=1)
    p = np.exp(eta)/(1+np.exp(eta))
    return np.random.rand(p.shape)<p

def evaluate(n_samples, n_features):
    sigma=0.1
    beta = np.random.rand(n_features+1)
    train_X, test_X = (np.random.standard_normal(size=(n_samples, n_features)) for _ in range(2))
    train_Y, test_Y  = (simulate_Y(X, beta, sigma) for X in (train_X, test_X))
    lin_model = LinearRegression().fit(train_X, train_Y)
    OLS_RSS = get_RSS(lin_model, test_X, test_Y)
    RSS = [OLS_RSS]
    for alpha in range(1, 31):
        RSS.append(evaluate_reg(0.1*alpha, train_X, test_X, train_Y, test_Y))

    return RSS

def get_RSS(model, X, Y):
    return model.score(X, Y)
    return np.sum((model.predict(X)-Y)**2)

def evaluate_reg(reg_param, train_X, test_X, train_Y, test_Y):
    model = Lasso(alpha=reg_param).fit(train_X, train_Y)    
    RSS = get_RSS(model, test_X, test_Y)# np.sum((model.predict(test_X)-test_Y)**2)/test_Y.size
    return RSS
    # print(reg_param, np.sum((model.predict(test_X)-test_Y)**2)/test_Y.size, np.sum(model.coef_>0))
    # print(reg_param, model.score(test_X, test_Y), model2.score(test_X, test_Y))

e = Evaluator()
simple_sum, score_sum = (0, 0)
n_sim = 10
for _ in range(n_sim):
    simple_score, scores = e.evaluate(100, 1000)
    simple_sum += simple_score
    score_sum += scores
score_sum /= 10
simple_sum /= 10
#rss_s = [e.evaluate(100, 1000) for _ in range(10)]
#print(np.sum(rss_s, axis=0)/10)
plt.plot(e.regularization_parameters, score_sum);
plt.axhline(simple_sum)
plt.show()
