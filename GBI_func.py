import numpy as np
import pandas as pd
from numba.typed import List
from math import ceil, sqrt, exp, log
from numba import njit, prange
import copy
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
from dateutil.relativedelta import relativedelta

trading_fee = 0.0015 # 매수 매도 15bp fee

m = 15# mu 포트폴리오 추출 갯수

@njit("(double(double))", cache=True, nogil=True)
def standard_normal_dist(x: float) -> float:
    return np.exp(-0.5*x**2)/(np.sqrt(2*np.pi))


@njit("(double, double, double[:], double, double[:], double[:], int32, double, double)", cache=True, nogil=True,
     )
def get_gbi_tree(W_init: float, G: float, cashflows: np.ndarray, T: float,
                 mu_portfolios: np.ndarray, sigma_portfolios: np.ndarray,
                 i_max_init: int, h: float, L: float) -> list:
    # Generate state space gridpoints
    # Initialize gridpoints with W(t=0)=W_init

    grid_points = List()

    grid_points.append(np.array([W_init]))

    # Create gridpoints for t=1,2,...,T

    sigma_max = sigma_portfolios[-1]

    mu_min, mu_max = mu_portfolios[0], mu_portfolios[-1]

    time_values = np.arange(0, int(T / h) + 1, 1)

    for tt in time_values[1:]:

        i_max_t = i_max_init * ceil(np.sqrt(tt * h))  # New i_max
        i_array_t = np.arange(-i_max_t, i_max_t + 1, 1)

        W_minus_i_max_prev = grid_points[tt - 1][0]  # Previus minimum wealth
        W_i_max_prev = grid_points[tt - 1][-1]  # Previous maximum wealth
        cashflow_prev = cashflows[tt - 1]  # Previous cashflow

        if W_minus_i_max_prev + cashflow_prev <= 0.:  # -> bankruptcy for previous minimum wealth

            # Search the minimum positive value of wealth such that W_i(t) + cashflow(t) > 0

            W_i_pos_prev = grid_points[tt - 1][grid_points[tt - 1] + cashflow_prev > 0.]

            # If no wealth is positive after taking into account the cashflow, bankruptcy is guaranteed
            assert len(W_i_pos_prev) != 0., 'Bankruptcy guaranteed'

            # Overwrite the previous minimum wealth
            W_minus_i_max_prev = W_i_pos_prev[0]

        # Compute the new minimum and maximum wealth values

        W_minus_i_max_t = (W_minus_i_max_prev + cashflow_prev) * exp(
            (mu_min - 0.5 * sigma_max ** 2) * h +
            sigma_max * sqrt(h) * (-3) * (sqrt(tt) - sqrt(tt - 1))
        )

        W_i_max_t = (W_i_max_prev + cashflow_prev) * exp(
            (mu_max - 0.5 * sigma_max ** 2) * h +
            sigma_max * sqrt(h) * 3 * (sqrt(tt) - sqrt(tt - 1))
        )

        # Generate the grid using interpolation
        grid_points_t = np.exp(
            ((i_array_t - (-i_max_t)) / (2. * i_max_t)) *
            (log(W_i_max_t) - log(W_minus_i_max_t)) +
            log(W_minus_i_max_t)
        )

        grid_points.append(grid_points_t)

    # Solve Bellman equation by backward recursion.

    # Value function at t=T

    value_i_t_plus_1 = np.where(grid_points[-1] >= G, 1., 0.)

    value_list = List()

    value_list.append(value_i_t_plus_1)

    mu_list = List()

    sigma_list = List()

    tp_matrix_list = List()

    # Start with t=T-1
    for tt in time_values[:-1][::-1]:

        transition_probabilities = np.zeros(
            shape=(grid_points[tt + 1].shape[0], grid_points[tt].shape[0])
        )

        tp_matrix_temp = np.zeros(
            shape=(grid_points[tt + 1].shape[0], grid_points[tt].shape[0])
        )
        
        #z_matrix = np.zeros(
        #    shape=(grid_points[tt+1].shape[0], grid_points[tt].shape[0])
        #)

        value_i_t = np.ones_like(grid_points[tt]) * -100.

        mu_i_t = np.zeros_like(grid_points[tt])

        sigma_i_t = np.zeros_like(grid_points[tt])

        # Estimate transition probabilities for each (mu,sigma) pair
        for sigma, mu in zip(sigma_portfolios, mu_portfolios):

            if sigma < 0.0001:
                sigma_inv = 1. / (0.0001)
            else:
                sigma_inv = 1. / sigma

            for j in prange(transition_probabilities.shape[0]):

                i_pos = np.argwhere(grid_points[tt] + cashflows[tt] > 0.)[0][0]

                for i in prange(i_pos, transition_probabilities.shape[1]):
                    z = (sigma_inv / sqrt(h)) * (
                            log(grid_points[tt + 1][j] / (grid_points[tt][i] + cashflows[tt])) -
                            (mu - 0.5 * sigma ** 2) * h
                    )

                    transition_probabilities[j, i] = standard_normal_dist(z)
                    #z_matrix[j,i] = abs(z)

            # Adjust zero-sum prob node
            #for i in range(transition_probabilities.shape[1]):
            #    if transition_probabilities[:,i].sum() == 0:
            #        transition_probabilities[z_matrix[:,i].argmin(),i] = 1
                    
            # Nomralize transition probabilities
            transition_probabilities = transition_probabilities / \
                                       transition_probabilities.sum(axis=0)

            # Obtain V(W_i)(t) for the given (mu,sigma) pair

            value_i_mu = value_i_t_plus_1 @ transition_probabilities
            mod_err_mask = (value_i_mu >= 1 - (1e-10))
            value_i_mu = np.where(mod_err_mask, 1, value_i_mu)
            mask = value_i_mu > value_i_t  # Check wether the new value with (mu, sigma) is greater than the previous one
            value_i_t = np.where(mask, value_i_mu, value_i_t)  # Update the maximum values

            # Update the mu and sigma values that maximizes V(W_i)
            mu_i_t = np.where(mask, mu, mu_i_t)

            sigma_i_t = np.where(mask, sigma, sigma_i_t)

            for opt in range(transition_probabilities.shape[1]):
                if mask[opt] == True:
                    tp_matrix_temp[:, opt] = transition_probabilities[:, opt]

        value_list.append(value_i_t)

        value_i_t_plus_1 = value_i_t  # Update V(W_i)(t+1)

        mu_list.append(mu_i_t)

        sigma_list.append(sigma_i_t)

        tp_matrix_list.append(tp_matrix_temp)

    mu_list = mu_list[::-1]

    sigma_list = sigma_list[::-1]

    tp_matrix_list = tp_matrix_list[::-1]

    value_list = value_list[::-1]

    return [value_list, mu_list, sigma_list, grid_points], tp_matrix_list


def cal_i_init(T: int, h: float, mu_min: float, mu_max: float, sigma_portfolios: np.ndarray):
    rho = 3

    sigma_max = sigma_portfolios[-1]

    sigma_min = sigma_portfolios[0]

    W_init = 100
    W_max_i = W_init
    W_min_i = W_init

    Total_W_gaps = []

    for i in range(int(T / h)):
        W_max_i = W_max_i * exp(
            (mu_max - 0.5 * sigma_max ** 2) * h * (i + 1) +
            sigma_max * sqrt(h) * 3 * sqrt(i + 1))

        W_min_i = W_min_i * exp(
            (mu_min - 0.5 * sigma_max ** 2) * h * (i + 1) +
            sigma_max * sqrt(h) * (-3) * sqrt(i + 1))

        if W_min_i < 0.01:
            W_min_i = 0.01
        
        Total_W_gaps.append(log(W_max_i / W_min_i))

    threshold = sigma_min / rho

    i_init = 25

    node_list = [2 * element + 1 for element in [i_init * ceil(sqrt(x)) for x in range(1, int(T / h) + 1, 1)]]

    while any(w / num > threshold for w, num in zip(Total_W_gaps, node_list)):
        if i_init == 100:
            break
        i_init += 5
        node_list = [2 * element + 1 for element in [i_init * ceil(sqrt(x)) for x in range(1, int(T / h) + 1, 1)]]

    return i_init


def cumprod(A_array=np.array([1]), B_array_list=[]):
    result = copy.deepcopy(A_array)
    for i in range(len(B_array_list)):
        result = result @ (B_array_list[i]).T

    return result


def get_bounds(stock_number, bound=(0, 0.3)):
    return (bound,) * stock_number


def get_constraints(characteristics=None):
    print(characteristics)
    if characteristics == None:
        stock_cap, stock_floor, bond_cap, bond_floor, alt_cap, alt_floor = tuple([None] * 6)
    elif characteristics == "Aggressive":
        stock_cap, stock_floor, bond_cap, bond_floor, alt_cap, alt_floor = (1.0, 0.3, 0.7, 0.0, 0.4, 0.0)

    elif characteristics == "Neutral":
        stock_cap, stock_floor, bond_cap, bond_floor, alt_cap, alt_floor = (0.7, 0.15, 0.85, 0.0, 0.4, 0.0)

    elif characteristics == "Conservative":
        stock_cap, stock_floor, bond_cap, bond_floor, alt_cap, alt_floor = (0.4, 0.0, 1.0, 0.0, 0.4, 0.0)
    else:
        pass

    return stock_cap, stock_floor, bond_cap, bond_floor, alt_cap, alt_floor


def get_mu(w, mu, sign=+1):
    return (w.T @ mu) * sign


def get_vol(w, cov_mat, sign=+1):
    return sqrt(w.T @ cov_mat @ w) * sign


def closest_value(input_list, input_value):
    arr = np.asarray(input_list)

    i = (np.abs(arr - input_value)).argmin()

    return i

def enlarge_constraints(constraints, new_con):
    constraints = list(constraints)
    constraints.append(new_con)
    return tuple(constraints)


def Cal_parameters(T, h, est_ret_data, class_tag, stock_N, start_dt, T_estimate, characteristics=None, w_range=(0, 0.3), restrictions="valid", tolerance=1e-10):
    # ind_i = est_ret_data.index.tolist().index(start_dt)

    estimate_date = est_ret_data.index[(est_ret_data.index.tolist().index(start_dt) - T_estimate + 1)]

    mean_returns_vector = (np.array((((est_ret_data.loc[estimate_date:start_dt] + 1).product()) **
                                     (252 / len(est_ret_data.loc[estimate_date:start_dt])) - 1).tolist(), order='C'))
    cov_matrix = np.array((est_ret_data.loc[estimate_date:start_dt].cov() * 252))

    bounds = get_bounds(stock_N, w_range)

    stock_cap, stock_floor, bond_cap, bond_floor, alt_cap, alt_floor = get_constraints(characteristics)

    cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]

    if (characteristics != None) and (restrictions == "valid"):
        cons.append({'type': 'ineq', 'fun': lambda x: stock_cap - sum(x[(class_tag.iloc[0, :] == "Stock").values])})
        cons.append({'type': 'ineq', 'fun': lambda x: sum(x[(class_tag.iloc[0, :] == "Stock").values]) - stock_floor})
        cons.append({'type': 'ineq', 'fun': lambda x: bond_cap - sum(x[(class_tag.iloc[0, :] == "Bond").values])})
        cons.append({'type': 'ineq', 'fun': lambda x: sum(x[(class_tag.iloc[0, :] == "Bond").values]) - bond_floor})
        cons.append({'type': 'ineq', 'fun': lambda x: alt_cap - sum(x[(class_tag.iloc[0, :] == "Alt").values])})
        cons.append({'type': 'ineq', 'fun': lambda x: sum(x[(class_tag.iloc[0, :] == "Alt").values]) - alt_floor})

    cons = tuple(cons)

    w_init = np.array([1 / stock_N] * stock_N)

    min_res = minimize(get_vol, w_init, args=(cov_matrix,), method="SLSQP", bounds=bounds, constraints=cons,
                       tol=tolerance)
    cons_min_vol = enlarge_constraints(cons, {'type': 'eq', 'fun': lambda x: get_vol(x, cov_matrix) - min_res.fun})
    mu_min_res = minimize(get_mu, w_init, args=(mean_returns_vector, -1,), method="SLSQP", bounds=bounds,
                          constraints=cons_min_vol, tol=tolerance)

    mu_min = get_mu(mu_min_res.x, mean_returns_vector)

    max_res = minimize(get_mu, w_init, args=(mean_returns_vector, -1,), method="SLSQP", bounds=bounds, constraints=cons,
                       tol=tolerance)
    cons_max_mu = enlarge_constraints(cons,
                                      {'type': 'eq', 'fun': lambda x: get_mu(x, mean_returns_vector) + max_res.fun})
    mu_max_res = minimize(get_vol, w_init, args=(cov_matrix,), method="SLSQP", bounds=bounds, constraints=cons_max_mu,
                          tol=tolerance)

    mu_max = get_mu(mu_max_res.x, mean_returns_vector)

    mu_portfolios = np.linspace(mu_min, mu_max, m)

    sigma_portfolios = []
    weight_portfolios = []

    for i in range(m):
        cons_adj = enlarge_constraints(cons, {'type': 'eq',
                                              'fun': lambda x: get_mu(x, mean_returns_vector) - mu_portfolios[i]})
        res = minimize(fun=get_vol, x0=w_init, args=(cov_matrix,)
                       , method="SLSQP", bounds=bounds, constraints=cons_adj, tol=tolerance)
        sigma_portfolios.append(get_vol(res.x, cov_matrix))
        weight_portfolios.append(res.x)

    sigma_portfolios = np.array(sigma_portfolios)
    weight_portfolios = np.array(weight_portfolios)

    i_max_0 = cal_i_init(T, h, mu_min, mu_max, sigma_portfolios)

    return mu_min, mu_max, mu_portfolios, sigma_portfolios, weight_portfolios, i_max_0


def get_bm_weight(est_ret_data, stock_ratio, stock_ticker, bond_ticker):
    ticker_list = est_ret_data.columns.tolist()
    weight_list = []
    
    for i in ticker_list:
        if i == stock_ticker:
            weight_list.append(stock_ratio)
        elif i == bond_ticker:
            weight_list.append(1-stock_ratio)
        else:
            weight_list.append(0)
    
    return np.array(weight_list)


def Cal_bm(est_ret_data, start_dt, T_estimate, stock_ratio, stock_ticker, bond_ticker):
    
    estimate_date = est_ret_data.index[(est_ret_data.index.tolist().index(start_dt) - T_estimate + 1)]
    
    mean_returns_vector = (np.array((((est_ret_data.loc[estimate_date:start_dt]+1).product())** 
                                     (252/len(est_ret_data.loc[estimate_date:start_dt])) -1).tolist(),order='C'))
    
    cov_matrix = np.array((est_ret_data.loc[estimate_date:start_dt].cov()*252))
    
    weight = get_bm_weight(est_ret_data, stock_ratio, stock_ticker, bond_ticker)
    
    return get_mu(weight,mean_returns_vector), get_vol(weight, cov_matrix), weight


def get_risk_profile_T(mu, sigma, T):
    mu_T_temp = mu * T
    sigma_T_temp = sigma * sqrt(T)

    first_moment = np.exp(mu_T_temp + ((sigma_T_temp) ** 2) / 2)
    second_moment = np.exp(2 * mu_T_temp + 2 * sigma_T_temp ** 2)

    return first_moment - 1, np.sqrt(second_moment - first_moment ** 2)


def calculate_wealth(multiple, goal, mu_array, sigma_array, horizon, frequency, portfolio_num, payment_type):
    periods = int(horizon * 1 / frequency)
    mod_goal = goal * multiple

    wealth_list = []
    fv_factors_list = []
    discount_factors_list = []

    if payment_type == "One_time":
        for i in range(portfolio_num):
            wealth_temp = mod_goal / (1 + get_risk_profile_T(mu_array[i], sigma_array[i], horizon)[0])
            wealth_list.append(wealth_temp)
        return np.array(wealth_list)

    elif payment_type == "Recurring":
        for i in range(portfolio_num):
            fv_factors = 0
            discount_factors = 0
            for j in range(periods):
                fv_factors += (get_risk_profile_T(mu_array[i], sigma_array[i], int((periods - j) * frequency))[0] + 1)
                discount_factors += 1 / (get_risk_profile_T(mu_array[i], sigma_array[i], int((j) * frequency))[0] + 1)
            fv_factors_list.append(fv_factors)
            discount_factors_list.append(discount_factors)
            wealth_temp = mod_goal / fv_factors
            wealth_list.append(wealth_temp)
        return np.array(wealth_list), np.array(discount_factors_list)


def calculate_prob(port_number, goal, wealth, mu_array, sigma_array, horizon, payment_type, discount_factors=1):
    mu, sigma = get_risk_profile_T(mu_array[port_number], sigma_array[port_number], horizon)

    return 1 - norm.cdf(goal / (discount_factors * wealth) - 1, mu, sigma)


def calculate_prob_opt(multiple, ind, goal, mu_array, sigma_array, horizon, frequency, payment_type, portfolio_num=m):
    wealth = calculate_wealth(multiple, goal, mu_array, sigma_array, horizon, frequency, portfolio_num, payment_type)

    mu, sigma = get_risk_profile_T(mu_array[ind], sigma_array[ind], horizon)

    return ((1 - norm.cdf(goal / (wealth[ind]) - 1, mu, sigma)) - 0.95) ** 2


def get_payment_range(mu_array, sigma_array, goal, horizon, frequency, payment_type, portfolio_num):
    
    ind = (np.array([get_risk_profile_T(mu_array[i], sigma_array[i], horizon)[0]/get_risk_profile_T(mu_array[i], sigma_array[i], horizon)[1]  for i in range(portfolio_num)])).argmax()

    multiple = 1

    wealth_arr = calculate_wealth(multiple, goal, mu_array, sigma_array, horizon, frequency, portfolio_num, "One_time")

    min_wealth = wealth_arr.min()
    min_port_num = wealth_arr.argmin()
    min_prob = calculate_prob(min_port_num, goal, min_wealth, mu_array, sigma_array, horizon, "One_time")

    con = ([{'type': 'ineq', 'fun':
        lambda x: goal * 0.95 -
                  calculate_wealth(x, goal, mu_array, sigma_array, horizon, frequency, portfolio_num, "One_time")[
                      ind]}])

    res = (minimize(calculate_prob_opt, multiple,
                    args=(ind, goal, mu_array, sigma_array, horizon, frequency, "One_time", portfolio_num,),
                    method="SLSQP", bounds=((1, 30),), constraints=con, tol=1e-10))

    max_multiple = res.x[0]

    wealth_arr = calculate_wealth(max_multiple, goal, mu_array, sigma_array, horizon, frequency, portfolio_num,
                                  "One_time")

    max_port_num = ind
    max_wealth = wealth_arr[max_port_num]
    max_prob = calculate_prob(max_port_num, goal, max_wealth, mu_array, sigma_array, horizon, "One_time")

    if payment_type == "One_time":
        return (min_wealth, min_port_num, min_prob, max_wealth, max_port_num, max_prob, max_multiple)

    elif payment_type == "Recurring":
        min_wealth, discount_factor = calculate_wealth(multiple, goal, mu_array, sigma_array, horizon, frequency,
                                                       portfolio_num, payment_type)
        min_wealth, discount_factor = min_wealth[min_port_num], discount_factor[min_port_num]

        min_prob = calculate_prob(min_port_num, goal, min_wealth, mu_array, sigma_array, horizon, payment_type,
                                  discount_factor)

        max_wealth, discount_factor = calculate_wealth(max_multiple, goal, mu_array, sigma_array, horizon, frequency,
                                                       portfolio_num, payment_type)
        max_wealth, discount_factor = max_wealth[max_port_num], discount_factor[max_port_num]

        max_prob = calculate_prob(max_port_num, goal, max_wealth, mu_array, sigma_array, horizon, payment_type,
                                  discount_factor)

        return (min_wealth, min_port_num, min_prob, max_wealth, max_port_num, max_prob, max_multiple)


def get_panel(res, col_ind):
    temp_data = []
    for i in range(len(res[0][1])):
        for j in range(len(res[0][1][i])):
            temp_data.append((i, np.log(res[0][3][i][j]), res[0][col_ind][i][j]))

    df = pd.DataFrame(np.array(temp_data), columns=["period", "wealth", "data"])
    df = df.pivot("wealth", "period", "data")

    return df.columns,df.index,df.values

def get_mu_distribution(res, col_ind,W0, G,Risktype, color_list = ['black','purple','red'],filename = ""):
    cmap = colors.LinearSegmentedColormap.from_list('',color_list)
    c, i, v = get_panel(res,col_ind)
    plt.pcolor(c, i, v,cmap=cmap)
    plt.colorbar()
    
    plt.hlines(y=np.log(W0), xmin=min(c), xmax=max(c), colors='aqua', linestyles='-', lw=2, label="ln(W0)")
    plt.hlines(y=np.log(G), xmin=min(c), xmax=max(c), colors='orange', linestyles='-', lw=2, label="ln(G)")
    plt.title("Mu_distribution")
    plt.xlabel("time_state_i")
    plt.ylabel("ln(wealth)")
    if Risktype == None:
        Risktype = "None"
    plt.savefig("figure/mu_dist_"+filename, facecolor = "white",bbox_inches='tight')
    plt.show()
    plt.close()
    
def expected_value(df_list, prob_df_list, ind, time_num, T, h, ref_list="pass", w_list = "pass"):
    
    T_num = int(T/h)
    
    if time_num == 1 :
        prob_array = prob_df_list[0].T
        
    else:
        prob_array = cumprod(prob_df_list[0].T,prob_df_list[1:time_num])
    
    if (0<ind<3) and (time_num ==T_num):
        return "Only expected value/wealth can be calculated for terminal time state"
    if time_num ==0:
        return "It is deterministic at initial time state"
    
    if ref_list != "pass":
        idx = [i for i in range(len(ref_list))]
        index2idx = dict(zip(ref_list,idx))
        mu_ind_array = np.array(list(map(lambda x: index2idx[x],df_list[ind][time_num])))
        
        index2idx_w = dict(zip(idx,w_list))
        w_array = np.array(list(map(lambda x: index2idx_w[x], mu_ind_array)))
        
        return (prob_array @ w_array)
    else:  
        prob_array = prob_array[0]
        return (prob_array * df_list[ind][time_num]).sum()    
    
def get_weight_transition(df_list, prob_df_list, ind, T, h, data, class_tag,Risktype, ref_list="pass", w_list = "pass",filename=""):
    temp_list = []
    for i in range(1,int(T/h)):
        temp_list.append(expected_value(df_list, prob_df_list, ind, i, T, h, ref_list, w_list)[0])
        
    weight_0 = w_list[(ref_list.tolist()).index(df_list[ind][0][0])]
    temp_list.insert(0,weight_0)
    
    df = pd.DataFrame(dict(zip(range(1,int(T/h)),temp_list)),index = data.columns)
    sec_weight_df = df
    # ax = df.T.plot.area(stacked = True)
    # ax.legend(loc = "center left",bbox_to_anchor = (1,0.5))
    # plt.title("Expected_Weight")
    # plt.xlabel("time_state_i")
    # plt.ylabel("weight")
    # if Risktype == None:
    #     Risktype = "None"
    # plt.savefig("figure/sec_weight_"+filename, facecolor = "white",bbox_inches='tight')
    # plt.show()
    
    
    df.loc[:,"tag"] = (class_tag.T).loc[df.index,0]
    df = df.set_index("tag")
    df = df.groupby(["tag"]).sum()
    df = df.loc[["Stock", "Bond", "Alt"], :]
    # ax = df.T.plot.area(stacked=True)
    # ax.legend(loc = "center left", bbox_to_anchor=(1,0.5))
    # plt.title("Expected_Weight")
    # plt.xlabel("time_state_i")
    # plt.ylabel("weight")
    # if Risktype == None:
    #     Risktype = "None"
    # plt.savefig("figure/class_weight_"+filename, facecolor = "white",bbox_inches='tight')
    # plt.show()
    #
    # plt.close()
    class_weight_df = df
    return sec_weight_df, class_weight_df
        
def get_expected_value_path(df_list, prob_df_list, ind, T, h,label_name,Risktype,
                            G = "pass", ref_list="pass", w_list = "pass", return_option = False,filename = ""):
    temp_list = []
    for i in range(1,int(T/h)):
        temp_list.append(expected_value(df_list, prob_df_list, ind, i, T, h, ref_list, w_list))
    
    if return_option == True:
        return temp_list
    else:
        if ind == 3:
            df = pd.DataFrame(np.array([temp_list,[G]*len(temp_list)]),columns = range(1,int(T/h)))
            df = df.T
            plt.plot(df.iloc[:,0],label = label_name[0])
            plt.plot(df.iloc[:,1],label = label_name[1])
            plt.title(label_name[0])
            plt.xlabel("time_state_i")
            plt.ylabel(label_name[0][9:])

        else:
            df = pd.DataFrame(temp_list,index = range(1,int(T/h)))
            plt.plot(df, label = label_name)   
            plt.title(label_name)
            plt.xlabel("time_state_i")
            plt.ylabel(label_name[9:])

        plt.legend(loc = "center left",bbox_to_anchor = (1,0.5))
        if Risktype == None:
            Risktype = "None"
        try:
            plt.savefig("figure/"+label_name+"_"+filename, facecolor = "white",bbox_inches='tight')
        except:
            plt.savefig("figure/"+label_name[0]+"_"+filename, facecolor = "white",bbox_inches='tight')
        plt.show()
        plt.close()
    
def draw_efficient_frontier(sigma_list,mu_list,Risktype,filename = ""):
    plt.plot(sigma_list,mu_list)
    plt.title("Efficient Frontier")
    plt.xlabel("volatility")
    plt.ylabel("return")
    if Risktype == None:
        Risktype = "None"
    plt.savefig("figure/Efficient_frontier_"+filename, facecolor = "white",bbox_inches='tight')
    plt.show()
    plt.close()
    
def get_percentile_wealth(df_list, prob_df_list, ind, time_num, T, h , percentile):
    # time_num >= 1
    
    if time_num == 1:
        prob_array = prob_df_list[0].T
    else:
        prob_array = cumprod(prob_df_list[0].T,prob_df_list[1:time_num])
    
    prob_array=prob_array[0]
    
    for i in range(len(df_list[ind][time_num])):
        if (prob_array)[:i].sum() < percentile:
            continue
        else:
            return df_list[ind][time_num][i]
            break

def get_percentile_wealth_path(df_list, prob_df_list, ind, T, h,percentile, yearly_smoothing = False):
    temp_list = []
    if yearly_smoothing == True:
        for i in range(int(1/h),int(T/h)+1,int(1/h)):
            temp_list.append(get_percentile_wealth(df_list,prob_df_list,ind,i,T,h,percentile))
    else:
        for i in range(1,int(T/h)+1):
            temp_list.append(get_percentile_wealth(df_list,prob_df_list,ind,i,T,h,percentile))
    return temp_list
    
def get_wealth_path(df_list, prob_df_list, ind, G, T, h, Risktype,  selected_wealth, bm_result, yearly_smoothing = False,filename=""):
    mu_path = (get_expected_value_path(df_list, prob_df_list, ind, T, h, "Expected_mu", Risktype, return_option = True))

    mu_path.insert(0, df_list[ind][0][0])
    
    #_10percentile_path = get_percentile_wealth_path(df_list, prob_df_list, 3, T, h,0.1,yearly_smoothing)
    _25percentile_path = get_percentile_wealth_path(df_list, prob_df_list, 3, T, h,0.25,yearly_smoothing)
    #_50percentile_path = get_percentile_wealth_path(df_list, prob_df_list, 3, T, h,0.50,yearly_smoothing)
    _75percentile_path = get_percentile_wealth_path(df_list, prob_df_list, 3, T, h,0.75,yearly_smoothing)
    #_90percentile_path = get_percentile_wealth_path(df_list, prob_df_list, 3, T, h,0.9,yearly_smoothing)

    _25percentile_path_bm = get_percentile_wealth_path(bm_result[0], bm_result[1], 3, T, h,0.25,yearly_smoothing)
    _75percentile_path_bm = get_percentile_wealth_path(bm_result[0], bm_result[1], 3, T, h,0.75,yearly_smoothing)
    
    time_axis = [i for i in range(0,int(T/h)+1)]
    time_axis_yearly = [i for i in range(0,int(T/h)+1,int(1/h))]
    
    if yearly_smoothing==True:
        time_percentile_axis = time_axis_yearly
    else:
        time_percentile_axis=time_axis
    
    wealth_path = [selected_wealth]    
    for i in range(len(mu_path)):
        wealth_path.append(wealth_path[-1] * (1+mu_path[i]*h))
    
    #_10percentile_path.insert(0,selected_wealth)
    _25percentile_path.insert(0,selected_wealth)
    #_50percentile_path.insert(0,selected_wealth)
    _75percentile_path.insert(0,selected_wealth)
    #_90percentile_path.insert(0,selected_wealth)
    
    _25percentile_path_bm.insert(0,selected_wealth)
    _75percentile_path_bm.insert(0,selected_wealth)
        
    # plt.plot(time_axis,wealth_path, color = 'black', zorder = 4)#)), label = "Expected_wealth")
    # #plt.plot(time_percentile_axis,_10percentile_path)#), label="Percentile_10")
    # #plt.plot(time_percentile_axis,_25percentile_path)
    # #plt.plot(time_percentile_axis,_50percentile_path)#), label="Percentile_50")
    # #plt.plot(time_percentile_axis,_75percentile_path)
    # #plt.plot(time_percentile_axis,_90percentile_path)#), label="Percentile_90")
    # plt.axhline(y=G, color='orange', linestyle='-',label = "Goal", zorder = 3)
    # plt.fill_between(time_percentile_axis,_75percentile_path_bm,_25percentile_path_bm,color = "lightpink" , zorder = 1)
    # plt.fill_between(time_percentile_axis,_75percentile_path,_25percentile_path,color = "skyblue" , zorder = 2)
    #
    #
    # plt.title("Expected_wealth"+("_Smoothing" if (yearly_smoothing == True) else ""))
    # plt.xlabel("time_state_i")
    # plt.ylabel("wealth")
    # #plt.legend(["Expected_wealth","Percentile_10","Percentile_25","Percentile_50","Percentile_75","Percentile_90"],loc = "center left",bbox_to_anchor = (1,0.5))
    # plt.legend(["Expected_wealth","Goal","BM_Port","Proposed_Port"],loc = "center left",bbox_to_anchor = (1,0.5))
    # if Risktype == None:
    #     Risktype = "None"
    # plt.savefig("figure/Expected_wealth_"+filename+("_Smoothing" if (yearly_smoothing == True) else ""), facecolor = "white",bbox_inches='tight')
    # plt.show()
    # plt.close()
    return time_percentile_axis, wealth_path