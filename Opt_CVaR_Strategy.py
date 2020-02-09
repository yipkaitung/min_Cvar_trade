#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Opt_CVaR_Strategy():
    
    @staticmethod
    def connect_to_IB(ip_address,socket_port,clientID):
        import ib_insync
        util.startLoop()
        global ib
        ib = IB()
        return ib.connect(ip_address, socket_port, clientId=clientID)
    
    @staticmethod # user should define portfolio, duration and barsize
    def read_data(stock_list,duration,barSize):
        import pandas as pd
        import numpy as np
        import ib_insync
        ib = IB()
        
        close_price={}
        price = []
        for stock in stock_list:
            contract = Stock(stock, 'SMART', 'USD')
            data = ib.reqHistoricalData(contract,endDateTime='',durationStr = duration, barSizeSetting=barSize,whatToShow='TRADES',useRTH=True,formatDate=1)
            df = util.df(data)
            price.append(df['close']) # get close price
        
        close_price = dict(zip(stock_list, price))
        df = pd.DataFrame(close_price)
        return df
    
    @staticmethod
    def log_return(df):
        log_return={}
        for symbol in enumerate(df):
            logreturn= np.log(df[symbol[1]])-np.log(df[symbol[1]].shift(1)) # log(Price_t) - log(Price_t-1) 
            log_return[symbol[1]] = logreturn
    
        df = pd.DataFrame(log_return)
        df=df.drop([0])
        df = df.reset_index(drop=True)
        return df
    
    @staticmethod
    def plot_cum_yield_curve(df):
        df.cumsum().plot(grid=True, figsize=(15,7))
         
    @staticmethod 
    def fit_best_dist(df):
        sns.pairplot(df,palette="Set2", diag_kind="kde",height=2.5) # plot KDE for each stock
        for symbol in enumerate(df):
            data = df[symbol[1]]
            distributions = [st.norm, st.t] #try to fit different distribution
            mles = []

            for distribution in distributions:
                f = distribution.fit(data)
                mle = distribution.nnlf(f, data)#calculate mle of different distribution
                mles.append(mle)

            results = [(distribution.name, mle) for distribution, mle in zip(distributions, mles)]
            best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0] # the smallest mle, the best-fit of stock
            print (symbol[1] + ' Best fit reached using {}, MLE value: {}'.format(best_fit[0].name, best_fit[1]))
            print ('Degree of freedom: {}, Mean: {}, Standard deviation: {}'.format(round(f[0]), f[1], f[2]))
            print()
            
    @staticmethod         
    def fit_copula(df):
        d = df.shape[1] # dimension of portfolio
        c_cop = ClaytonCopula(dim=d)
        model = c_cop.fit(df,method='ml') #using ml to estimate theta
        print(model.summary())
        return model
     
    @staticmethod 
    def gen_U(model,size):
        sample=model.random(size) # generate random U
        sample = pd.DataFrame(sample)
        return sample
    
    @staticmethod
    def est_return_rate(U,df):
        
        means={}
        for symbol in enumerate(df):
            means[symbol[1]] = df[stock].mean()
        
        sigmas={}
        for symbol in enumerate(df):
            sigmas[symbol[1]] = df[stock].std()
    
        
        dfs={}
        for symbol in enumerate(df):
            degree=st.t.fit(df[symbol[1]], floc = df[symbol[1]].mean(), fscale=df[symbol[1]].std())[0]
            degree = round(degree)
            dfs[symbol[1]] = degree
    
        r_est={}
        for symbol in enumerate(df):
            n = U.shape[0]
            u = U[symbol[0]]
            mu = means[symbol[1]]
            sigma = sigmas[symbol[1]]
            degree = dfs[symbol[1]]
            r_est[symbol[1]] = st.t.ppf(u,loc=mu,scale=sigma,df=degree)
    
        df_stock_return_est = pd.DataFrame(r_est)
        return df_stock_return_est
    
    class hist_sim():
        
        
        @staticmethod
        def find_opt_weight(df,alpha,num_sim):
            def histCVaR(histSeries,n,alpha):
                histSeries = pd.DataFrame(histSeries)
                histSeries.sort_values([histSeries.columns[0]],ascending=True,inplace=True)
                histSeries['count'] = np.arange(1,n+1)
                m = int(n*alpha)
                histSeries = histSeries[histSeries['count'] <= m]
                return -histSeries[histSeries.columns[0]].sum()/m

            def pfCVaR(weights):
                return histCVaR(pd.DataFrame(np.mat(df)*np.mat(weights).T, columns = ['portVaR']).portVaR,df.shape[0], 0.01)
            
            
            def genrate_random_weight(num_sim,df):
    
                stock_weights0 = []
                num_portfolios = num_sim
                d = df_stock_price.shape[1]
    
                for single_portfolio in range(num_sim):
                    weights0 = np.random.random(d)
                    weights0 /= np.sum(weights0)
                    stock_weights0.append(weights0)
    
                portfolio0 = {}
                for counter,symbol in enumerate(df):
                    portfolio0[symbol+' weight'] = [weight0[counter] for weight0 in stock_weights0]

                df_random_weight = pd.DataFrame(portfolio0)
                column_order = [stock+' weight' for stock in df]
                df_random_weight = df_random_weight[column_order]
                return df_random_weight
            
            def cal_ES(df_random_weight,df_stock_return_est,alpha):
    
                weight_returns = np.dot(df_random_weight, df_stock_return_est.T)
                weight_returns.sort(axis = 1)
                tail = int(alpha*df_stock_return_est.shape[0])
                weight_returns_tail = weight_returns[:,:tail]
                Cvar = -np.mean(weight_returns_tail, axis = 1)
                return Cvar
            
            
            cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
            res = opt.minimize(pfCVaR, [1/df.shape[1]]*df.shape[1], constraints= cons,bounds= [(0,1)]*df.shape[1], method = 'SLSQP')
            min_cvar = res['fun']
            
            if min_cvar <= 0.01:
                df_random_weight = genrate_random_weight(num_sim,df) #generate 1000 random weight
                Cvars = cal_ES(df_random_weight,df_stock_return_est,alpha)
                
                candidate=np.where(Cvars<=0.01)
                p=candidate[0]
                Cvars[p]

                returnallocation=np.zeros(len(p))
                for i in range (len(p)):
                  j=p[i]
                  returnallocation[i] = np.dot(df_random_weight.loc[j,:], df_stock_return_est.mean())

                list_returnallocation = returnallocation.tolist()
                max_index = list_returnallocation.index(max(list_returnallocation))
                opt_weight = df_random_weight.loc[p[max_index],:]
                opt_cvar = Cvars[p[max_index]]
                print('The optimal weight are:')
                print(opt_weight)
                print()
                print('The optimal Expected Shortfall is:')
                print(opt_cvar)
            else:
                print('The minmum ES not meet requirement, please re-arrange your stock choice.' )
            
            return opt_weight
        
    @staticmethod   
    def backtest(opt_weight,benchmark):
        def cal_opt_port_return(opt_weight):
            his_sim_weight = []
            for weight in opt_weight:
                his_sim_weight.append([weight])
            hist_return_rate = np.array(df_stock_return)
            Hist_Sim_Port = np.dot(hist_return_rate, his_sim_weight)
            Hist_Sim_Port=pd.DataFrame(Hist_Sim_Port)
            return Hist_Sim_Port

        def polt_backtest(benchmark,Hist_Sim_Port):
    
            benchmark_index = read_data(benchmark,duration,barSize)
            index_return = log_return(benchmark_index)
    
            becktest = pd.concat( [index_return[benchmark],Hist_Sim_Port], axis=1)
            becktest.columns = ['MSCI','Hist_Sim_Port']
            becktest['MSCI'].cumsum().plot(grid=True, figsize=(15,7),color='r',label="MSCI")
            becktest['Hist_Sim_Port'].cumsum().plot(grid=True, figsize=(15,7),color='y',label="Hist_Sim_Port")
            plt.legend()
        
        Hist_Sim_Port= cal_opt_port_return(opt_weight)
        polt_backtest(benchmark,Hist_Sim_Port)
        
    @staticmethod
    def trading(stock_list,opt_weight,principal):
        now_price = np.array(df_stock_price)[-1]
        opt_weight = np.array(opt_weight).T
        wei_money =  opt_weight*principal
        wei_money=wei_money[0]
        num_share_trade= wei_money/now_price
    
        stock_buy = dict(zip(stock_list, num_share_trade))
                     
        for key, value in stock_buy.items():
            contract = Stock(key, 'SMART', 'USD') 
            ib.qualifyContracts(contract)
            order = MarketOrder('BUY', int(value))
            trade = ib.placeOrder(contract, order)
            print(trade)
            print()

