from pprint import pprint
from time import time
import matplotlib.pyplot as plt
from scipy import interpolate
import os
from datetime import datetime
from matplotlib import colors
import pickle
from GBI_func import *
from flask import Flask
from flask_cors import CORS
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

def read_pickle(file_nm):
    with open('data/{}.pickle'.format((file_nm)), 'rb') as file:
        df = pickle.load(file)
    return df
def save_pickle(df, file_nm):
    with open('data/{}.pickle'.format((file_nm)), 'wb') as file:
        pickle.dump(df, file, protocol = pickle.HIGHEST_PROTOCOL)

data = read_pickle("Data_rev")
class_tag = read_pickle("Sec_Tagging_rev")

stock_N = len(data.columns)

ret_data_origin = data / data.shift(1) - 1
ret_data_origin = ret_data_origin.dropna()
est_ret_data = copy.deepcopy(ret_data_origin)

ret_data_origin["Year+Month"] = [x[:7] for x in ret_data_origin.index.astype(str)]
ret_data_origin["Date"] = ret_data_origin.index.values

today_Y, today_m, today_d = 2022, 4, 29
today = pd.Timestamp(today_Y, today_m, today_d)

# G = 200. * 1000000  # 목표 금액
T = 10.  # 은퇴 시기
Payment_type = "One_time"  # 거치식: One_time / 적립식: Recurring
# L = 1.0 * G

h = 1 / 12  # 리밸런싱 주기(default 1/12)
period = int(12 * h)

weight_range = (0, 0.3)

T_estimate = ret_data_origin.index.tolist().index(today) + 1

def match_risk_type(num):
    print(num)
    num = int(num)
    if num<=2:
        return 'Conservative', ['Conservative']
    elif num<=3:
        return 'Neutral', ['Conservative', 'Neutral']
    else:
        return 'Aggressive', ['Conservative', 'Neutral', 'Aggressive']
@app.route('/', methods=['GET', 'POST'])
def check():

    return "FLASK IS RUNNING"

@app.route('/make_port_by_type/<Risk_type>', methods=['GET', 'POST'])
def make_port_by_type(Risk_type, h=h, T=10, weight_range=weight_range):
    Risk_type, _ = match_risk_type(Risk_type)
    mu_list, sigma_list, portw_list, i_max_init = Cal_parameters(T, h, est_ret_data, class_tag, stock_N, today,
                                                                 T_estimate, characteristics=Risk_type, w_range=weight_range,
                                                                 restrictions="valid", tolerance=1e-9)[2:]
    save_pickle(list(mu_list), 'port_type/mu_list_{}'.format(Risk_type))
    save_pickle(list(sigma_list), 'port_type/sigma_list_{}'.format(Risk_type))
    save_pickle(list(portw_list), 'port_type/portw_list_{}'.format(Risk_type))
    save_pickle(i_max_init, 'port_type/i_max_init_{}'.format(Risk_type))
    return {"message": "made_port_sucessfully"}

@app.route('/get_init_range/<Risk_type>_<G>_<T>', methods=['GET', 'POST'])
def get_init_range(G, T, Risk_type, h=h, Payment_type=Payment_type):
    T = int(T)
    G = int(G) * 10000
    Risk_type, _ = match_risk_type(Risk_type)

    mu_list, sigma_list, portw_list, i_max_init = Cal_parameters(T, h, est_ret_data, class_tag, stock_N, today,
                                                                 T_estimate, characteristics=Risk_type, w_range=weight_range,
                                                                 restrictions="valid", tolerance=1e-9)[2:]
    minimum_wealth, minportnum, minprob, maximum_wealth, maxportnum = get_payment_range(mu_list, sigma_list, G, T, h,
                                                                                        Payment_type, m)[:5]
    diff = maximum_wealth - minimum_wealth
    quater1 = minimum_wealth + diff/4
    quater2 = minimum_wealth + diff/4 * 2
    quater3 = minimum_wealth + diff/4 * 3
    quater4 = maximum_wealth
    return {"min": int(minimum_wealth//10000), "max": int(maximum_wealth//10000),
            "quater1":quater1//10000, "quater2":quater2//10000, "quater3":quater3//10000, "quater4":quater4//10000}

@app.route('/get_init_port/<Risk_type>_<G>_<T>_<W0>', methods=['GET', 'POST'])
def get_init_port(G, T, Risk_type, W0, weight_range=weight_range):
    T = int(T)
    G = float(G) * 10000
    W0 = float(W0) * 10000
    all_Risk_types = ['Conservative', 'Neutral', 'Aggressive']
    Port_Risk_type, Risk_types = match_risk_type(Risk_type)
    ports = dict()
    for Risk_type in all_Risk_types:
        port_per_risk = dict()
        if Risk_type not in Risk_types:
            port_per_risk['goal_prob'] = "-"
            port_per_risk['loss_prob'] = "-"
            port_per_risk['return'] = "-"
        else:

            mu_list, sigma_list, portw_list, i_max_init = Cal_parameters(T, h, est_ret_data, class_tag, stock_N, today,
                                                                         T_estimate, characteristics=Risk_type,
                                                                         w_range=weight_range, restrictions="valid",
                                                                         tolerance=1e-9)[2:]

            minimum_wealth, minportnum, minprob, maximum_wealth, maxportnum = get_payment_range(mu_list, sigma_list, G, T, h,
                                                                                                Payment_type, m)[:5]

            wealths = [np.exp(np.log(minimum_wealth) + i / 4 * np.log(maximum_wealth / minimum_wealth)) for i in range(5)]

            print(datetime.now())
            selected_wealth = W0


            filename = "None" if Risk_type == None else Risk_type

            L = 1.0 * selected_wealth

            res_main = get_gbi_tree(selected_wealth, G, np.array([0.] * int(T / h)), T, mu_list,
                                    sigma_list, i_max_init, h, L)

            mu_bm, sigma_bm, w_bm = Cal_bm(est_ret_data, today, T_estimate, 0.6, "ACWI US Equity", "TLT US Equity")

            res_bm = get_gbi_tree(selected_wealth, G, np.array([0.] * int(T / h)), T, np.array([mu_bm] * 2),
                                  np.array([sigma_bm] * 2), i_max_init, h, L)


            sec_weight_df, class_weight_df = get_weight_transition(df_list=res_main[0], prob_df_list=res_main[1], ind=1, T=T, h=h,
                                  data=data, class_tag=class_tag, Risktype=Risk_type, ref_list=mu_list,
                                  w_list=portw_list, filename=filename)

            time_percentile_axis, wealth_path = get_wealth_path(df_list=res_main[0], prob_df_list=res_main[1], ind=1, G=G, T=T, h=h,
                            Risktype=Risk_type, selected_wealth=selected_wealth, bm_result=res_bm, yearly_smoothing=False,
                            filename=filename)

            recomm_prob = res_main[0][0][0][0]
            recomm_prob_loss = sum(cumprod(res_main[1][0].T,
                                           res_main[1][1:])[0][:closest_value(res_main[0][3][-1], L)])
            recomm_er = ((cumprod(res_main[1][0].T, res_main[1][1:])[0] * res_main[0][3][-1]).sum() / selected_wealth) ** (
                        1 / T) - 1
            print(
                f"제안 포트폴리오 \n\n목표달성확률: {recomm_prob * 100:.1f}%\n원금손실률: {recomm_prob_loss * 100:.1f}%\n기대수익률: {recomm_er * 100:.1f}%")

            print(datetime.now())
            port_per_risk['goal_prob'] = round(recomm_prob * 100, 2)
            port_per_risk['loss_prob'] = round(recomm_prob_loss * 100, 2)
            port_per_risk['return'] = round(recomm_er * 100, 2)

            del sec_weight_df['tag']
            init_port = sec_weight_df[[sec_weight_df.columns[0]]]
            init_port[sec_weight_df.columns[0]] = init_port[sec_weight_df.columns[0]].apply(lambda x: int(round(x,0)))
            init_port *= 100
            init_port = init_port[init_port[sec_weight_df.columns[0]]!=0]
            init_port_list = list()

            for count, idx in enumerate(init_port.index):
                port = dict()
                port['name'] = idx
                port['y'] = init_port.loc[idx].values[0]
                port['z'] = init_port.loc[idx].values[0]
                init_port_list.append(port)

                table = {
                    "id": str(count+1),
                    "name": str(idx),
                    "info": str(W0*init_port.loc[idx].values[0])+'원',
                    "description": str(idx),
                    "date": {
                        "progress": '29',
                        "type": 'success',
                    },
                    "status": {
                        "progress": '29',
                        "type": 'bar-gray-light',
                    },
                }
            sec_weight_df = sec_weight_df.reset_index().melt(id_vars='index', value_vars=sec_weight_df.columns)
            sec_weight_df['value'] = sec_weight_df['value'].apply(lambda x: int(round(x*100, 0)))
            sec_weight_df['variable'] = sec_weight_df['variable'].apply(lambda x: 'T'+str(x))
            # sec_weight_df=sec_weight_df[sec_weight_df['value']!=0]
            sec_weight_list = sec_weight_df[['variable', 'value', 'index']].values.tolist()

            port_per_risk['sec_weight_df'] = sec_weight_list
            port_per_risk['sec_weight_tickers'] = list(set(sec_weight_df['index'].tolist()))
            port_per_risk['init_port'] = init_port_list

            big_ratio = []
            for idx in class_weight_df.mean(axis=1).index:
                ratio = dict()
                ratio['name'] = idx
                ratio['value'] = round(class_weight_df.mean(axis=1).loc[idx] *100,2)
                big_ratio.append(ratio)
            port_per_risk['class_weight_df'] = big_ratio
            port_per_risk['wealth_path_idx'] = ["T"+str(i) for i in range(len(wealth_path))]
            port_per_risk['wealth_path'] = wealth_path
            port_per_risk['wealth_path_goal'] = [G for i in range(len(wealth_path))]
            port_per_risk['wealth_path_goal_step'] = [W0+(i+1)*(G-W0)/len(wealth_path) for i in range(len(wealth_path))]

        ports[Risk_type] = port_per_risk


    return ports


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    print(1)