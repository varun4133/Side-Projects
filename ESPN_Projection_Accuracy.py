import requests
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Setting Paramaters
espn_s2   = ('AEAVE3tAjA%2B4WQ04t%2FOYl15Ye5f640g8AHGEycf002gEwr1Q640'
            'iAvRF%2BRYFiNw5T8GSED%2FIG9HYOx7iwYegtyVzOeY%2BDhSYCOJr'
            'CGevkDgBrhG5EhXMnmiO2GpeTbrmtHmFZAsao0nYaxiKRvfYNEVuxr' 
            'CHWYewD3tKFa923lw3NC8v5qjjtljN%2BkwFXSkj91k2wxBjrdaL5P' 
            'p1Y77%2FDzQza4%2BpyJq225y4AUPNB%2FCKOXYF7DTZ5B%2BbuH'
            'fyUKImvLaNJUTpwVXR74dk2VUMD9St')
league_id = 1382012
swid      = '{E01C2393-2E6F-420B-9C23-932E6F720B61}'


      
positionmap = {0 : 'QB', 2 : 'RB', 4 : 'WR',6 : 'TE', 
               16: 'Def', 17: 'K', 23: 'Flex'}
positions=[0,2,4,6,16,17]
weeks= range(2,3)

seasons= [2019]

#Scraping Data from ESPN and entering it into DF


def Web_Scrape(season,week):
    url_req = ('https://fantasy.espn.com/apis/v3/games/ffl/seasons/' +
          str(season) + '/segments/0/leagues/' + str(league_id) +
          '?view=mMatchup&view=mMatchupScore')  
    
    params={'scoringPeriodId': week ,'competitorlimit':30}
    cookies={"SWID": swid, "espn_s2": espn_s2}
    req=requests.get(url_req,params=params,cookies=cookies)
    data=req.json()
    return data
    


def Create_DF(season,weeks,positions):
    result={}
    for ind,week in enumerate(weeks):
        data=Web_Scrape(season,week)
        weekstr='week '+str(week)
        for position in positions:
            for fantasy_team in data['teams']:       
                for player in fantasy_team['roster']['entries']:
                    if player['lineupSlotId']==position:
                        player_name = player['playerPoolEntry']['player']['fullName']
                        for stat in player['playerPoolEntry']['player']['stats']:
                            if stat['scoringPeriodId'] != week:
                                continue
                            if stat['statSourceId'] == 0:
                                actual_score = stat['appliedTotal']
                            elif stat['statSourceId'] == 1:
                                proj_score = stat['appliedTotal']
                        if result.get(player_name)==None:
                            result[player_name]={}
                            result[player_name]['Position']=positionmap[position]
                        if proj_score!=0 and actual_score!=0:
                            pct_diff=((proj_score-actual_score)/actual_score)*100
                            if pct_diff >500 or (pct_diff<500 and actual_score<0):
                                continue
                            result[player_name][weekstr+' Projection']= proj_score    
                            result[player_name][weekstr+' Actual']= actual_score
                            result[player_name][weekstr]=pct_diff                 
        df=pd.DataFrame.from_dict(result,orient='index')
        return df                           

def Create_DF_Dict(seasons,weeks,positions):
    dfs={}
    for season in seasons:
        df=Create_DF(season,positions,weeks)
        dfs[season]=df                            
    return dfs

#Data Processing

dfs=Create_DF_Dict(seasons,positions,weeks)

weeklist=[]
for week in weeks:
    weeklist.append('week '+str(week))




def DF_Manipulation(seasons):
    
    df_player_dict={}
    df_position_averaged_dict={}
    for season in seasons:
        df=dfs[season]
        #individual player stats
        df_player=df.reset_index().melt(id_vars=['index','Position'],value_vars=weeklist,var_name='Week',value_name='Percent Difference').set_index('index')
        df_player.index.name='Name'
        df_player_dict[season]=df_player
        #grouped by position and week, averaged over players
        df_position_averaged=df_player.groupby(['Position','Week'],as_index=False)['Percent Difference'].mean()
        df_position_averaged['Season']=season
        df_position_averaged_dict[season]=df_position_averaged   
    return df_player_dict,df_position_averaged_dict

_,df_position_averaged_dict=DF_Manipulation(seasons)        
    


def df_combination(df_position_averaged_dict):
    dflist=[df_position_averaged_dict[key] for key in df_position_averaged_dict]
    combined_df=pd.concat(dflist)
    
    seasontrend_df=combined_df.groupby(['Position','Season'])['Percent Difference'].agg([np.mean,np.std]).reset_index()
    seasontrend_df['std']=abs(seasontrend_df['std']/seasontrend_df['mean'])
    seasontrend_df.columns=['Position','Season','Mean Percent Difference','Normalized Variation in Percent Difference']
    
    total_df=combined_df.groupby('Position')['Percent Difference'].agg([np.mean,np.std]).reset_index()
    total_df['std']=abs(total_df['std']/total_df['mean'])
    total_df.columns=['Position','Mean Percent Difference','Normalized Variation in Percent Difference']
    return seasontrend_df,total_df

seasontrend_df,total_df=df_combination(df_position_averaged_dict)

#Plotting



#Plotting vs week in season (to see if projections get btter as season carries out)   
fig=plt.figure()
sns.set_style("whitegrid", {'axes.grid' : False})
titlestring='Season 2019 Projection Accuracy'
ax=sns.scatterplot(x='Week',y='Percent Difference',data=df_position_averaged_dict[2019],hue='Position')
plt.ylabel('Mean Percent Difference ((Proj-Actual)/Actual)*100')
plt.title(titlestring)



#comparing between positions and seasons for many weeks
fig=plt.figure()

sns.set_style("whitegrid", {'axes.grid' : False})
titlestring='Seasons '+str(min(seasons))+' - ' +str(max(seasons))      
ax=sns.scatterplot(x='Season',y='Mean Percent Difference',data=seasontrend_df,hue='Position')

ax.set_xticks(seasons)
ax.set(xticklabels=[str(s) for s in seasons])  
ax.set(xlim=(min(seasons)-1, max(seasons)+1))
plt.title(titlestring)
plt.ylabel('Mean Percent Difference ((Proj-Actual)/Actual)*100')

plt.show()

#comparing between positions for many seasons/weeks    

fig=plt.figure()
titlestring='Seasons '+str(min(seasons))+' - ' +str(max(seasons))    
ax=sns.scatterplot(x='Normalized Variation in Percent Difference',y='Mean Percent Difference',data=total_df,hue='Position')
plt.title(titlestring)
sns.set_style("whitegrid", {'axes.grid' : False})
plt.ylabel('Mean Percent Difference ((Proj-Actual)/Actual)*100')
plt.show()

