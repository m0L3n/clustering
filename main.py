import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sns as sns
from sqlalchemy import create_engine
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

#import data
sales = pd.read_csv("C:/Users/lenzm/Desktop/data_auto/sales.csv")
claims = pd.read_csv("C:/Users/lenzm/Desktop/data_auto/claims.csv")
car_details = pd.read_csv("C:/Users/lenzm/Desktop/data_auto/car_details.csv")

# Overview data=
    # car_details:
        car_details.info()
        car_details.isna().sum()
        car_details.select_dtypes('int64').nunique()
    #create car df without duplicates
    car = car_details.drop_duplicates()
    car = car.reset_index(drop=True) #reset the index.
    # sales:
        sales.info()
        sales.isna().sum()
        sales.select_dtypes('object').nunique()
        sales.select_dtypes('int64').nunique()
        sales_nonduplicat = sales.drop_duplicates()
        sales_nonduplicat
    #claims
        claims.info()
        claims.isna().sum()

    #load data to postgresql (to perform exercise 2)
        engine = create_engine('postgresql+psycopg2://postgres:******@localhost:****/postgres')
        sales_nonduplicat.to_sql('sales',engine, if_exists='replace' )
        claims.to_sql('claims',engine, if_exists='replace')
        car.to_sql('car_details',engine, if_exists='replace')  #I dropped duplicates, so I don't have to do it in sql.

# DATA combined car_id, sales and total claims amount. In order to use this combined data set for the K Prototype analysis.
    #sum up total amount of claim per car in claims table
    claims_total = pd.merge(claims,sales[['car_id','sell_price']], how='left', left_on=['car_id'],right_on=['car_id'] )
    claims_total['claim_total'] = 0
    for i in claims_total.index:
        if claims_total.iloc[i,6] =='closed_fully_processed':
            if claims_total.iloc[i,2] == 1:
                claims_total.iloc[i,10] = claims_total.iloc[i,9]
            if claims_total.iloc[i,2] == 0:
                if claims_total.iloc[i,3] == 1 and claims_total.iloc[i,7] == 'yes':
                    claims_total.iloc[i,10] = (claims_total.iloc[i,4] + claims_total.iloc[i,8])
                if claims_total.iloc[i,3] == 1 and claims_total.iloc[i,7] == 'no':
                    claims_total.iloc[i,10] = claims_total.iloc[i,4]
    claims_total.describe()
    #merge aggregated claims, sales and car in one df.
    df_merge = pd.merge(car['car_id '],sales[['sell_price', 'car_id']], how='left', left_on=['car_id '],right_on=['car_id']).drop('car_id ', axis=1)
    df_sales_claims = pd.merge(df_merge,claims_total[['car_id','claim_total']], how='left', left_on=['car_id'],right_on=['car_id']) #.drop('car_id', axis=1)
    df_total = pd.merge(df_sales_claims,car, how='inner', left_on=['car_id'],right_on=['car_id '])   #.drop('car_id ','road_worthy', axis=1)
    df_total = df_total.drop_duplicates().reset_index()
    df_total.fillna({'claim_total':0}, inplace = True)
    df_total = df_total.drop(['car_id ','car_id','road_worthy'], axis=1)

###########################################  -- K Prototypes  --  #############################################################################
# DATA K Prototype
    df_proto = df_total
    df_array = df_proto.values
    cost = []

    # fit k prototype; find optimal k:
        for i in range(1,10,1):
            try:
                kprototype = KPrototypes(n_jobs=-1, n_clusters=i, init='Huang', random_state=0)
                kprototype_cluster = kprototype.fit(df_array, categorical=[2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                cost.append(kprototype.cost_)
            except:
                break
        #plot cost for each k (for elbow method)
        y = np.array([i for i in range(1,10,1)])
        x = np.array(cost)
        plt.plot(y,x)
    # Using the elbow method on the ploted graph, the amount of clusters should be k=3. Check 3,4 and 5 clusters.
        #k=3
            kprototype = KPrototypes(n_jobs=-1, n_clusters=3, init='Huang', random_state=42)
            kprototype_cluster = kprototype.fit(df_array, categorical=[2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            cost.append(kprototype.cost_)
            #Cluster Centroids:
            cluster_centroids_proto = pd.DataFrame(kprototype_cluster.cluster_centroids_)
            cluster_centroids_proto.columns = df_proto.columns
            cluster_centroids_proto.rename(columns={'ac_type':'car_preowner_count','fuel_type':'ac_type','gear_type':'fuel_type','car_preowner_count':'gear_type'}, inplace=True)
                #k=4
                    kprototype4 = KPrototypes(n_jobs=-1, n_clusters=4, init='Huang', random_state=0)
                    kprototype_cluster4 = kprototype4.fit(df_array, categorical=[2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                    cost.append(kprototype4.cost_)
                    #Cluster Centroids:
                    cluster_centroids_proto4 = pd.DataFrame(kprototype_cluster4.cluster_centroids_)
                    cluster_centroids_proto4.columns = df_proto.columns
                    cluster_centroids_proto4.rename(columns={'ac_type':'car_preowner_count','fuel_type':'ac_type','gear_type':'fuel_type','car_preowner_count':'gear_type'}, inplace=True)
                #k=5
                    kprototype5 = KPrototypes(n_jobs=-1, n_clusters=5, init='Huang', random_state=0)
                    kprototype_cluster5 = kprototype5.fit(df_array, categorical=[2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                    cost.append(kprototype5.cost_)
                    # Cluster Centroids:
                    cluster_centroids_proto5 = pd.DataFrame(kprototype_cluster5.cluster_centroids_)
                    cluster_centroids_proto5.columns = df_proto.columns
                    cluster_centroids_proto5.rename( columns={'ac_type': 'car_preowner_count', 'fuel_type': 'ac_type', 'gear_type': 'fuel_type','car_preowner_count': 'gear_type'}, inplace=True)

    #add cluster number to data frame.
    cluster_numbers_proto = pd.DataFrame(kprototype_cluster.labels_)
    cluster_numbers_proto.columns = ['cluster']
    clusterDF_proto = pd.concat([df_proto,cluster_numbers_proto], axis=1).reset_index()
    clusterDF_proto = clusterDF_proto.drop(['level_0','index'], axis=1)
        #cluster 0  #overview Clusters:
        cluster_0_proto = clusterDF_proto[clusterDF_proto['cluster']==0].reset_index()
        cluster_0_proto = cluster_0_proto.drop('index', axis=1)
        cluster_0_proto.info()
        cluster_0_proto.describe()
        #cluster 1
        cluster_1_proto = clusterDF_proto[clusterDF_proto['cluster']==1].reset_index()
        cluster_1_proto = cluster_1_proto.drop('index', axis=1)
        cluster_1_proto.info()
        cluster_1_proto.describe()
        #cluster 2
        cluster_2_proto = clusterDF_proto[clusterDF_proto['cluster']==2].reset_index()
        cluster_2_proto = cluster_2_proto.drop('index', axis=1)
        cluster_2_proto.info()
        cluster_2_proto.describe()

    #correlation by features
    correlation = clusterDF_proto.corr()
    correlation_list = []
    for i in range(0, len(correlation.index)):
        for j in range(0, len(correlation.columns)):
            if ((correlation.iloc[i,j] > 0.5 or correlation.iloc[i,j] < -0.5) and correlation.iloc[i,j] != 1):
                correlation_list.append(str(correlation.index[i] + ' and ' + correlation.columns[j] + ' : ' + str(round(correlation.iloc[i,j], 2))))
    correlation_list = '\n'.join(correlation_list)
    print(correlation_list)
    #conclusion from clusters
    clusterDF_proto.groupby('cluster').agg(['median','mean']).T

    #plot =
        #plot a boxplot for numerical features:
            # df for plots
            temp_df = clusterDF_proto.loc[:, ['sell_price', 'claim_total', 'cluster']]
            temp_0 = temp_df[temp_df['cluster'] == 0].reset_index().rename( columns={'sell_price': 'sell_price_0', 'claim_total': 'claim_total_0', 'cluster': 'cluster_0'})
            temp_1 = temp_df[temp_df['cluster'] == 1].reset_index().rename(columns={'sell_price': 'sell_price_1', 'claim_total': 'claim_total_1', 'cluster': 'cluster_1'})
            temp_2 = temp_df[temp_df['cluster'] == 2].reset_index().rename(columns={'sell_price': 'sell_price_2', 'claim_total': 'claim_total_2', 'cluster': 'cluster_2'})
            temp_merge = pd.concat([temp_1, temp_0], axis=1)
            temp_final = pd.concat([temp_merge, temp_2], axis=1)
            print(temp_final.describe())
            # boxplots for sell price and claims.
            boxplot_sell = temp_final.boxplot(column=['sell_price_0', 'sell_price_1', 'sell_price_2'])
            plt.title('sell price per cluster')
            boxplot_claims = temp_final.boxplot(column=['claim_total_0', 'claim_total_1', 'claim_total_2'])
            plt.title('claims per cluster')

        #cathegorical features:
            #plot boolean features:
            col_list_bool= ['has_diesel_particulate_filter','has_airbags','has_alarm_system','has_abs','has_metallic_color','has_tuning']
            for col in col_list_bool:
                clusters = ['cluster 1','cluster 2','cluster 3']
                bool_1 = []
                bool_0 = []
                x = np.arange(len(clusters))
                width = 0.35
                fig, ax = plt.subplots()
                for i in range(0,kprototype.n_clusters,1):
                    bool_1.append(clusterDF_proto.loc[clusterDF_proto['cluster'] == i, col].value_counts().loc[1])
                    bool_0.append(clusterDF_proto.loc[clusterDF_proto['cluster'] == i, col].value_counts().loc[0])
                rects1 = ax.bar(x - width / 2, bool_1, width, label='value 1')
                rects2 = ax.bar(x + width / 2, bool_0, width, label='value 0')
                ax.set_ylabel('quantity')
                ax.set_title('Feature ' + col + ' per Cluster')
                ax.set_xticks(x, clusters)
                ax.legend()
                ax.bar_label(rects1, padding=3)
                ax.bar_label(rects2, padding=3)
                fig.tight_layout()
                plt.show()
            #print categorical features
            col_list_cate = ['ac_type', 'fuel_type', 'gear_type', 'outside_colour', 'xenon_light', 'navigation_system', 'radio_system']
            for col in col_list_cate:
                for i in range(0, kprototype.n_clusters, 1):
                    print('Cluster ' +str(i)+ ' and Category '+ col )
                    print(clusterDF_proto.loc[clusterDF_proto['cluster'] == i, col].value_counts())