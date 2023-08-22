import multiprocessing
import time as TIME
from switch import Organization, Stake
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from math import sqrt
from multiprocessing import Pool
from time import time, sleep
import nkpack as nk
import pandas as pd
from scipy.stats import skewnorm
import math
import random
from datetime import date
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from datetime import datetime, date

reciprocal = "off"
'''Modeler'''
T = 200  # time steps to observe in an organization
MC = 1000  # number of simulation runs
cnf = 1.96   # 2.576

simulationDataSetup = pd.read_excel("simulationParameters.xlsx")
print(simulationDataSetup)

listOfSimulations = [simulationCounter for simulationCounter in range(len(simulationDataSetup))]
simulationIds = []
def my_function(simulationCounter):
    currentID = simulationDataSetup.iloc[simulationCounter].at["ID"]
    if currentID in simulationIds:
        return
    simulationIds.append(currentID)
    comment = str(currentID)
    RHO = 0
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()  # take date

    '''landscapes'''
    P = 2  # number of landscapes (Environment and organization), in this model always equals 2
    N = 8  # number of bits for each landscape
    K = int(simulationDataSetup.iloc[simulationCounter].at["K"])  # number of internal interdependencies
    C = int(simulationDataSetup.iloc[simulationCounter].at["C"])  # number of coupled bits
    S = 1  # number of coupled peers
    '''Stakeholders'''
    eff = 1  # Level of Effort (Effort for all stakeholders is equal)
    ham = 2  # the radius that stakeholders look for their strategies(hamming distance)
    Num = int(simulationDataSetup.iloc[simulationCounter].at["NumStake"])  # number of stakeholders included in the process
    skew = simulationDataSetup.iloc[simulationCounter].at[
        "Skewness"]  # skewness of stakeholders self_interest  distribution (correlations)
    stake_err_env = 0  # stakeholders Errors about environment location
    stake_err_per = 0  # stakeholders Errors about performance of the strategy
    Vote_err = stake_err_per  # stakeholders Errors while voting

    '''Organization open'''
    '''shocks'''
    shock = simulationDataSetup.iloc[simulationCounter].at["Shock"]
    sw = "on"  # switch shocks
    ShockTime = simulationDataSetup.iloc[simulationCounter].at["ShockTime"]
    annealing = 'off'
    comment += f'SA={annealing}'
    react = 'on'
    voting_activate = 'on'  # if it is 'on' stakeholders can vote eventhough they havent made any idea
    sat_implement = "off"
    print("sat implement=", sat_implement)
    '''Environment'''
    environment_change_rate = float(simulationDataSetup.iloc[simulationCounter].at["NumStake"] / 100)
    print('environment_change_rate', environment_change_rate)
    env_meth = "random"

    Crf_err = 0  # Error while evaluating performance of proposals
    Env_err = 0

    '''Organization closed'''
    closed_per_err = 0  # Organization Error about the performance

    for i in range(1):
        '''A list of random initial positions for Environment'''
        environment_positions = []
        for i in range(MC):
            environment = np.random.choice(2, N)
            environment_positions.append(environment)

        '''A list of random initial positions for Organization and stakeholders'''
        organization_positions = []
        for j in range(MC):
            pos = np.random.choice(2, N)
            organization_positions.append(pos)

        Correlations = []
        for _ in range(MC):
            random_numbers = skewnorm.rvs(skew, size=int(Num) + 2)
            random_numbers = 2 * ((random_numbers - np.min(random_numbers)) /
                                  (np.max(random_numbers) - np.min(random_numbers))) - 1

            random_numbers_filtered = random_numbers[(random_numbers != 1) & (random_numbers != -1)]
            if len(random_numbers_filtered) != Num:
                for _ in range(Num - len(random_numbers_filtered)):
                    random_numbers_filtered.append(0)
            Correlations.append(random_numbers_filtered.tolist())
        """Errors"""
        S_e_bs = [stake_err_env for i in range(Num)]  # stakeholders errors about environment
        S_e_ps = [stake_err_per for i in range(Num)]  # stakeholders errors about performance of the strategies
        Efforts = [eff for i in range(Num)]  # efforts

    def single_iteration(Env_iteration,Org_pos_iteration,correlations,correlations2):#One complete simulation for T time steps
        
        #define different organizations (every one starts from a same position and environment is same for all of them)
        firm1 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name='firm1')        
        firm2 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name = 'firm2')
        firm3 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name = 'firm3')
        firm4 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name = 'firm4')                 
        
        
        #Switch reciprocal mode
        firm1.net = reciprocal
        firm2.net = reciprocal
        firm3.net = reciprocal
        firm4.net = reciprocal

        
        #define organization and environmet landscapes
        # define landscape for one of them and copy it for the rest
        firm1.define_tasks() 
        firm2.nature = firm1.nature
        firm3.nature = firm1.nature 
        firm4.nature = firm1.nature
        
        #Time steps
        firm1.T = T 
        firm2.T = T
        firm3.T = T
        firm4.T = T
        
        ##remove stakeholders who havent proposed
        firm1.voting_activate = voting_activate
        firm2.voting_activate = voting_activate
        firm3.voting_activate = voting_activate
        firm4.voting_activate = voting_activate
        
        if sw == "on":
             firm1.time_shock = T-50
             firm2.time_shock = T-50
             firm3.time_shock = T-50
             firm4.time_shock = T-50
             '''shocks'''
             firm1.shock(shock)
             firm2.shocked_nature = firm1.shocked_nature
             firm3.shocked_nature = firm1.shocked_nature
             firm4.shocked_nature = firm1.shocked_nature
       
        else:
            firm1.time_shock = T+50
            firm2.time_shock = T+50
            firm3.time_shock = T+50
            firm4.time_shock = T+50
       
        # Including stakeholders
        firm1.hire_stakeholders()
        firm2.hire_stakeholders()
        firm3.hire_stakeholders()
        firm4.hire_stakeholders()
        
        #hamming distance in the neighborhood
        firm1.ham = ham
        firm2.ham = ham
        firm3.ham = ham
        firm4.ham = ham
        
        # Annealing simulation status
        firm1.annealing = annealing
        firm2.annealing = annealing
        firm3.annealing = annealing
        firm4.annealing = annealing

        # time represents time_step
        for time in range(1,T):  
            
            #Phase1: Idea generation
            firm1.idea_generation(time,ham)
            firm2.idea_generation(time,ham)
            firm3.idea_generation(time,ham)
            firm4.idea_generation(time,ham)
            
            
            #Phase2: Strategy selection
            vote_err = Vote_err # stakeholders' Error while voting 
            org_craft_err = 0 # organization's Error
            firm1.crafting_stk_env(Crf_err,time)
            firm2.closed(0,0,eff,ham,time)
            firm3.tidemanf(Vote_err,time)
            firm4.closed(0,0,eff,ham,time)
            

            #phase3: strategy implemetation
            if sat_implement == "on":
                firm1.implement_strategy(time)
                firm2.implement_strategy(time)
                firm3.implement_strategy(time)
                firm4.implement_strategy(time)
            elif sat_implement == "off":
                firm1.implementation(time)
                firm2.implementation(time)
                firm3.implementation(time)
                firm4.implementation(time)
            
            
            #stakeholders' reaction
            if react == 'on':
                firm1.stake_feedback(1, time)
                firm2.stake_feedback(1, time)
                firm3.stake_feedback(1, time)
                firm4.stake_feedback(1, time)
            
            #Environment change'''
            speed = 1
            firm1.environment_change_random(speed,environment_change_rate,time)
            firm2.env = firm1.env
            firm3.env = firm1.env
            firm4.env = firm1.env
     
                
        # Elicit results for firm1
        bit_states1 = []
        for dec_state in firm1.strategy_line[0, :]:
            bit_states1.append(firm1.dec_to_bin(dec_state))
        performances1 = []
        t1 = 0
        for bit_state in bit_states1:
            if t1 < firm1.time_shock:
                performance = firm1.nature.phi(None, bit_state)[0]
            else:
                performance = firm1.shocked_nature.phi_shock(None, bit_state)[0]
            performances1.append(performance)
            t1 += 1
        
        # Elicit results for firm2
        bit_states2 = []
        for dec_state in firm2.strategy_line[0, :]:
            bit_states2.append(firm2.dec_to_bin(dec_state))
        performances2 = []
        t2 = 0
        for bit_state in bit_states2:
            if t2 < firm2.time_shock:
                performance = firm2.nature.phi(None, bit_state)[0]
            else:
                performance = firm2.shocked_nature.phi_shock(None, bit_state)[0]
            performances2.append(performance)
            t2 += 1
        
        # Elicit results for firm3
        bit_states3 = []
        for dec_state in firm3.strategy_line[0, :]:
            bit_states3.append(firm3.dec_to_bin(dec_state))
        performances3 = []
        t3 = 0
        for bit_state in bit_states3:
            if t3 < firm3.time_shock:
                performance = firm3.nature.phi(None, bit_state)[0]
            else:
                performance = firm3.shocked_nature.phi_shock(None, bit_state)[0]
            performances3.append(performance)
            t3 += 1
        
        # Elicit results for firm4
        bit_states4 = []
        for dec_state in firm4.strategy_line[0, :]:
            bit_states4.append(firm4.dec_to_bin(dec_state))
        performances4 = []
        t4 = 0
        for bit_state in bit_states4:
            if t4 < firm4.time_shock:
                performance = firm4.nature.phi(None, bit_state)[0]
            else:
                performance = firm4.shocked_nature.phi_shock(None, bit_state)[0]
            performances4.append(performance)
            t4 += 1
        
        sat1 = firm1.satisfaction()
        sat2 = firm2.satisfaction()
        sat3 = firm3.satisfaction()
        sat4 = firm4.satisfaction()
        
        return performances1, performances2, performances3, performances4, sat1, sat2, sat3, sat4

            
    ##############################################################################################
    results1 = [] 
    results2 = []
    results3 = []
    results4 = []
    SAT1 = []
    SAT2 = []
    SAT3 = []
    SAT4 = []
    Correlations_fixed = None
    '''Run model for MC times'''
    for mc in range(MC):
         result1 , result2, result3, result4, sat1, sat2, sat3, sat4 = single_iteration(environment_positions[mc], organization_positions[mc],Correlations[mc],Correlations[mc])
         results1.append(result1)
         results2.append(result2)
         results3.append(result3)
         results4.append(result4)
         
         SAT1.append(sat1)
         SAT2.append(sat2)
         SAT3.append(sat3)
         SAT4.append(sat4)
         percentage = int((mc + 1) / MC * 100)
         # Print only unique percentages to avoid clutter
         if mc == 0 or percentage > int(mc / MC * 100):
            print(f"Progress: {percentage}%")

    #convert the results into numpy arrays    
    arr_results1 = np.array(results1)
    arr_results2 = np.array(results2)
    arr_results3 = np.array(results3)
    arr_results4 = np.array(results4)

    arr_SAT1 = np.array(SAT1)
    arr_SAT2 = np.array(SAT2)
    arr_SAT3 = np.array(SAT3)
    arr_SAT4 = np.array(SAT4)

    #confidence levels
    output1 = np.reshape(arr_results1,(MC,T))
    average_performance1 = np.mean(output1,0)
    s1 = np.std(output1,0)
    confidence_value1 = cnf*(s1/math.sqrt(MC))

    output2 = np.reshape(arr_results2,(MC,T))
    average_performance2 = np.mean(output2,0)
    s2 = np.std(output2,0)
    confidence_value2 = cnf*(s2/math.sqrt(MC))

    output3 = np.reshape(arr_results3,(MC,T))
    average_performance3 = np.mean(output3,0)
    s3 = np.std(output3,0)
    confidence_value3 = cnf*(s3/math.sqrt(MC))

    output4 = np.reshape(arr_results4,(MC,T))
    average_performance4 = np.mean(output4,0)
    s4 = np.std(output4,0)
    confidence_value4 = cnf*(s4/math.sqrt(MC))

    ###################################################

    outputSAT1 = np.reshape(arr_SAT1,(MC,T))
    average_sat1 = np.mean(outputSAT1,0)

    t10 = np.std(outputSAT1,0)
    confidence_value10 = cnf*(t10/math.sqrt(MC))
    #

    outputSAT2 = np.reshape(arr_SAT2,(MC,T))
    average_sat2 = np.mean(outputSAT2,0)

    t20 = np.std(outputSAT2,0)
    confidence_value20 = cnf*(t20/math.sqrt(MC))
    #
    outputSAT3 = np.reshape(arr_SAT3,(MC,T))
    average_sat3 = np.mean(outputSAT3,0)

    t30 = np.std(outputSAT3,0)
    confidence_value30 = cnf*(t30/math.sqrt(MC))
    #
    outputSAT4 = np.reshape(arr_SAT4,(MC,T))
    average_sat4 = np.mean(outputSAT4,0)

    t40 = np.std(outputSAT4,0)
    confidence_value40 = cnf*(t40/math.sqrt(MC))
    ####

    '''writing to files'''
    today_str = str(today)
    current_time_str = str(current_time)
    current_time_str = current_time_str.replace(":","&")
    parent_dir = r"C:\Users\sabanihashen\Desktop\MilanResults"
    path = os.path.join(parent_dir,comment, today_str, current_time_str)
    os.makedirs(path,exist_ok = True)
    np.save(f'{path}\K={K} C={C} SKEW={skew} MC={MC} Num={Num} S_e_bs={S_e_bs[0]} S_e_ps={S_e_ps[0]} Crf_err={Crf_err},Vote_err={Vote_err} shock={shock} tideman',output1)
    np.save(f'{path}\K={K} C={C} SKEW={skew} MC={MC} Num={Num} S_e_bs={S_e_bs[0]} S_e_ps={S_e_ps[0]} Crf_err={Crf_err},Vote_err={Vote_err} shock={shock} selection',output2)
    np.save(f'{path}\K={K} C={C} SKEW={skew} MC={MC} Num={Num} S_e_bs={S_e_bs[0]} S_e_ps={S_e_ps[0]} Crf_err={Crf_err},Vote_err={Vote_err} shock={shock} closed',output3)
    np.save(f'{path}\K={K} C={C} SKEW={skew} MC={MC} Num={Num} S_e_bs={S_e_bs[0]} S_e_ps={S_e_ps[0]} Crf_err={Crf_err},Vote_err={Vote_err} shock={shock} tidemanf',output4)




    # Set a consistent style for clarity
    plt.style.use('classic') 

    # Font settings suitable for APA
    font = {'family': 'Times New Roman',
            'color':  'black',
            'weight': 'normal',
            'size': 12,
            }

    time = [t for t in range(0, T)]

    # Create a function for plotting to reduce repetitive code
    def plot_data(time, avg_performance, confidence_value, color, linestyle, label, marker, marker_spacing):
        plt.plot(time, avg_performance, color=color, linestyle=linestyle, label=label, marker=marker, markersize=5, markevery=marker_spacing)
        plt.fill_between(time, avg_performance - confidence_value, avg_performance + confidence_value, alpha=0.1, color=color)

    # Plot data using different colors, line styles, and markers for differentiation
    plot_data(time, average_performance1, confidence_value1, "blue", '-', "Idea Generation phase is Opened", "o", 10)
    #plot_data(time, average_performance2, confidence_value2, "green", '--', "Idea generation and strategy selectio phases are opened", "s", 10)
    plot_data(time, average_performance3, confidence_value3, "red", '-.', "Idea Generation and Strategy Selection(limited) phases are Opened", "^", 10)
    plot_data(time, average_performance4, confidence_value4, "purple", ':', "Closed Strategy Making", "d", 10)

    # Adding labels with APA recommended font size and style
    plt.xlabel("Time", fontdict=font)
    plt.ylabel("Performance", fontdict=font)
    plt.legend(loc="best", fontsize='small')

    # Add a grid to the plot
    plt.grid(True)

    # Instead of a title, use a descriptive caption below the figure when incorporating it into your paper.
    # If you must provide source details, add it as text on the plot
    plt.figtext(0.5, 0, f"Figure. K={K}, C={C}, SKEW={skew}, MC={MC}, Num={Num} Shock={shock}. Source: [Milan]", ha="center", fontsize=10, wrap=True)

    # Show the plot
    plt.tight_layout() # Adjusts subplot params for better layout
    plt.savefig(f'{path}\\{comment}.pdf')
    plt.savefig(f'{path}\\{comment}.jpg')
    plt.show()
    return



# Ensure you have the definition of the 'my_function' and other required functions and classes here
# ...

# Creating a list of simulation counters
listOfSimulations = [simulationCounter for simulationCounter in range(len(simulationDataSetup))]
simulationIds = []

if __name__ == '__main__':
    # Create a pool of workers
    with multiprocessing.Pool() as pool:
        # Call my_function with multiple arguments in parallel and show progress bar
        results = list(tqdm(pool.imap(my_function, listOfSimulations), total=len(listOfSimulations)))
