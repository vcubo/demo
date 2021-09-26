
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px


# FUNCTIONS #
## Decomposition of impacts considering the model and its coefficients


def partials (df, df_coef, df_part_index):
    df_part = pd.DataFrame([df_coef['COUNTRY']*(df['COUNTRY_RMEAN']),
                           df_coef['LOB']*(df['LOB_RMEAN']),
                           df_coef['SITE']*(df['SITE_RMEAN']),
                           df_coef['PSIZE']*(df['PSIZE_RMEAN']),
                           df_coef['CSIZE']*(df['CSIZE_RMEAN']),
                           df_coef['SOC'] * df['SOC_EMEAN']* (df['SOC']-df['SOC_MIT'] *df_coef['MIT_ef']),
                           df_coef['PROC']*df['PROC_EMEAN']*(df['PROC']-df['PROC_MIT']*df_coef['MIT_ef']),
                           df_coef['ENG'] * df['ENG_EMEAN']* (df['ENG']-df['ENG_MIT'] *df_coef['MIT_ef']),
                           df_coef['WEA'] * df['WEA_EMEAN']* (df['WEA']-df['WEA_MIT'] *df_coef['MIT_ef']),
                           df_coef['MGM'] * df['MGM_EMEAN']* (df['MGM']-df['MGM_MIT'] *df_coef['MIT_ef']),
                          ],
                          index = df_part_index
                         ).transpose()
    #please find a more elegant way to get this df!:
    df_part_median = pd.DataFrame([df_part.median().transpose().tolist(),
                                 df_part_index, ['Uncertainty']*5+['Risk']*5]
                                 ).transpose()
    df_part_median.columns=['Impact', 'Variable','Factor']

    RAN_median = df['DEV_RAN'].median()
    EVE_median = df['DEV_EVE'].median()
    subt_uncert = df_part_median[df_part_median['Factor']=='Uncertainty']['Impact'].sum()
    subt_risk =  df_part_median[df_part_median['Factor']=='Risk']['Impact'].sum()

    df_part_median['Impact'][df_part_median['Factor']=='Uncertainty']= df_part_median['Impact'][df_part_median['Factor']=='Uncertainty']*RAN_median/subt_uncert
    if subt_risk != 0:
        df_part_median['Impact'][df_part_median['Factor']=='Risk'] = df_part_median['Impact'][df_part_median['Factor']=='Risk']*EVE_median/subt_risk

    return df_part_median

## Statistics function (mean, median, etc)
decimals = 4 #decimals shown in results
def df_stats(df):
        #List of statistics for DEV_RAN(uncertainty median deviation), DEV_EVE(risks' median deviation) and DEV_TOT (total deviation median)
    DEV_mean = [np.round(np.mean(df['DEV_RAN']),decimals), np.round(np.mean(df['DEV_EVE']),decimals), np.round(np.mean(df['DEV_TOT']),decimals)]
    DEV_median = [np.round(np.median(df['DEV_RAN']),decimals), np.round(np.median(df['DEV_EVE']),decimals), np.round(np.median(df['DEV_TOT']),decimals)]
    factor = [1+DEV_median[0],1+DEV_median[1]]  #deviation caused by uncertainty (0) and by risks (1)
    #Mean duration deviation (in months) and partial
    #DUR_delta_mean = np.mean(df['DUR_AC']-df['DUR_BL'])
    #DUR_delta_median = np.median(df['DUR_AC']-df['DUR_BL'])
    #DUR_delta_comp = [DUR_delta_median/factor[0], DUR_delta_median/factor[1]]
    results_dict = {'median':DEV_median, 'means':DEV_mean, 'factors': factor}
    return results_dict

## FILTER-list generator FUNCTION
def filter_gen(selection, df):
    filter_list = [i and j and k and l and m for i, j, k, l, m in
                      zip((df['COUNTRY'] == selection[0])^(selection[0]== 'All'),
                          (df['LOB'] == selection[1])^(selection[1] == 'All'),
                          (df['SITE'] == selection[2])^(selection[2] == 'All'),
                          (df['PR_SIZE'] == selection[3])^(selection[3] == 'All'),
                          (df['MC_SIZE'] == selection[4])^(selection[4] == 'All'))]
    return filter_list

## HISTOGRAM AND BAR CHART GENERATOR
def const_figures(df_base,df_comp, hist_xbins_size, df_coef, df_part_index):
    partials_df_comp = partials(df_comp, df_coef, df_part_index)
    figh1 = go.Histogram(x=df_base['DEV_TOT'], opacity=0.7, name='Total deviation',xbins={"size": hist_xbins_size})
    figh2 = go.Histogram(x=df_base['DEV_RAN'], opacity=0.5, name='Uncertainty',xbins={"size": hist_xbins_size/2})
    figh3 = go.Histogram(x=df_base['DEV_EVE'], opacity=0.5, name='Risk events impact',xbins={"size": hist_xbins_size/2})
    impact_deco = go.Bar(x=partials_df_comp['Factor'],y=partials_df_comp['Impact'])
    ## FILTERED HISTOGRAMS
    figh1f = go.Histogram(x=df_comp['DEV_TOT'], opacity=0.7, name='Total deviation <br>-selected projects',xbins={"size": hist_xbins_size})
    figh2f = go.Histogram(x=df_comp['DEV_RAN'], opacity=0.5, name='Uncertainty ',xbins={"size": hist_xbins_size/2})
    figh3f = go.Histogram(x=df_comp['DEV_EVE'], opacity=0.5, name='Risk events impact ',xbins={"size": hist_xbins_size/2})
    ## COMPOSED DEVIATION DISTRIBUTION
    g_dev_hist1 = go.FigureWidget(data=[figh1,figh1f],
                                 layout=go.Layout(#title=dict(text="Total composed uncertainty and risk's impact distribution", x = 0),
                                                                   barmode='overlay',
                                                                   bargap = 0.01,
                                                                   xaxis=dict(tickformat=".0%",
                                                                             title="Deviation"),
                                                                  yaxis=dict(title="Projects"),
                                                                   legend=dict(yanchor="top",
                                                                               y=0.99,
                                                                               xanchor="left",
                                                                               x=0.675),
                                                                   margin=dict(b=40, t=30,l=40))
                                                                   #plot_bgcolor ='#000000')
                                    )
    ## DECOMPOSED UNCERTAINTY/RISK IMPACTS DEVIATION DISTRIBUTION                             )
    g_dev_hist2 = go.FigureWidget(data=[figh3f, figh2f],
                                  layout=go.Layout(#title=dict(text="Decomposed uncertainty and risks' distributions", x = 0),
                                                                    barmode='overlay',
                                                                    bargap = 0.01,
                                                                    xaxis=dict(tickformat=".0%",
                                                                              title="Deviation"),
                                                                   yaxis=dict(title="Projects"),
                                                                    legend=dict(yanchor="top",
                                                                                y=0.99,
                                                                                xanchor="left",
                                                                                x=0.63),
                                                                    margin=dict(b=40, t=30,l=40))
                                 )
    ## DECOMPOSED UNCERTAINTY/RISK IMPACTS DEVIATION MEDIANS
    dev_comp_bar = px.bar(partials_df_comp,x='Factor',y='Impact', color = 'Variable').update_layout(#{'paper_bgcolor': 'whitesmoke'},
                                  yaxis=dict(tickformat=".1%"),
                                  #height=130,
                                  #paper_bgcolor='whitesmoke',
                                  #title=dict(text="Uncertainty and risk's decomposition (medians)", x=0),
                                  margin=dict(b=40, t=50,l=40)
                                 )
    #subt_uncert = str(partials_df_comp[partials_df_comp['Factor']=='Uncertainty'].sum())
    #dev_comp_bar.add_annotation( # add a text callout with arrow
    #    text=subt_uncert, x="Uncertainty", y=0.18, arrowhead=1, showarrow=True)
    return [g_dev_hist1,g_dev_hist2,dev_comp_bar]

## DISTRIBUTION FITTING
def fit_distr(df, hist_xbins_size):
    '''Generates lognormal pdf and cdf fitting total deviation data'''
    main_param_c1 = sp.stats.lognorm.fit(df['DEV_TOT'])
    x = np.linspace(0,1,int(1/hist_xbins_size))
    lognorm_pdf = sp.stats.lognorm.pdf(x,main_param_c1[0],main_param_c1[1], main_param_c1[2])
    lognorm_cdf = sp.stats.lognorm.cdf(x,main_param_c1[0],main_param_c1[1], main_param_c1[2])

    main_pdf_c1 = (lognorm_pdf)
    main_cdf_c1 = (lognorm_cdf)
    # HISTOGRAM + FIT
    figh1 = go.Histogram(x=df['DEV_TOT'], opacity=0.7, name='Total deviation',xbins={"size": hist_xbins_size})
    g_hist_fit = go.FigureWidget(data=[figh1],
                              layout=go.Layout(
                                  #title=dict(text='Deviation distribution and Lognormal fit'),
                                  barmode='overlay',
                                  #paper_bgcolor='whitesmoke',
                                  #plot_bgcolor='slategray'
                                  bargap = 0.01,
                                  xaxis=dict(tickformat=".1%"),
                      margin=dict(b=40, t=30,l=40))
                              )
    scale = len(df['DEV_TOT'])/(lognorm_pdf.sum())
    g_hist_fit.add_scatter(y = main_pdf_c1*scale, x = x, name = 'Lognormal fit pdf')
    # FIT PDF AND CDF
    #create an empty histogram to superpose pdf and cdf
    hist_dumb = go.Histogram(x=np.zeros(len(df['DEV_TOT'])), opacity=0.0, name='',xbins={"size": hist_xbins_size})
    g_pdf_cdf = go.FigureWidget(data=[hist_dumb]*0,
                              layout=go.Layout(
                                  #title=dict(text='Deviation distribution and Lognormal fit'),
                                  barmode='overlay',
                                  #paper_bgcolor='whitesmoke',
                                  #plot_bgcolor= 'ghostwhite',#'slategray'
                                  bargap = 0.01,
                                  xaxis=dict(tickformat=".1%"),
                                  yaxis=dict(tickformat=".0%"),
                      margin=dict(b=40, t=30,l=40))
                              )
    g_pdf_cdf.add_scatter(y = main_pdf_c1/np.max(main_pdf_c1), x = x, name = 'Lognormal fit pdf<br>(100% = mode)')
    g_pdf_cdf.add_scatter(y = lognorm_cdf, x = x, name = 'Lognormal fit cdf')
    return [g_hist_fit, g_pdf_cdf, main_param_c1, scale*main_pdf_c1, main_pdf_c1/np.max(main_pdf_c1), main_pdf_c1*scale]

## CALCULATION OF PARTIAL IMPACTS -BY VARIABLES AND RISKS EVENTS
def fit_probs(list):
    total = list.sum()
    sum = 0
    sum2 = 0
    len_list = len(list)
    for i in range(len_list):
        sum += list[i]
        if sum/total >= 0.5:
            p50 = (i+1)/(len_list)
            break
    for j in range(len_list):
        sum2 += list[j]
        if sum2/total >= 0.8:
            p80 = (j+1)/(len_list)
            break
    return (p50, p80, len_list, sum, total)

def compute_partials (df, df_part_index, df_coef):
    df[df_part_index[0]] = df_coef['COUNTRY']*(df['COUNTRY_RMEAN'])
    df[df_part_index[1]] = df_coef['LOB']*(df['LOB_RMEAN'])
    df[df_part_index[2]] = df_coef['SITE']*(df['SITE_RMEAN'])
    df[df_part_index[3]] = df_coef['PSIZE']*(df['PSIZE_RMEAN'])
    df[df_part_index[4]] = df_coef['CSIZE']*(df['CSIZE_RMEAN'])
    df[df_part_index[5]] = df_coef['SOC']*df['SOC_EMEAN']*(df['SOC']-df['SOC_MIT'])*df_coef['MIT_ef']
    df[df_part_index[6]] = df_coef['PROC']*df['PROC_EMEAN']*(df['PROC']-df['PROC_MIT'])*df_coef['MIT_ef']
    df[df_part_index[7]] = df_coef['ENG']*df['ENG_EMEAN']*(df['ENG']-df['ENG_MIT'])*df_coef['MIT_ef']
    df[df_part_index[8]] = df_coef['WEA']*df['WEA_EMEAN']*(df['WEA']-df['WEA_MIT'])*df_coef['MIT_ef']
    df[df_part_index[9]] = df_coef['MGM']*df['MGM_EMEAN']*(df['MGM']-df['MGM_MIT'])*df_coef['MIT_ef']
    df['SOC (NM)'] = (df['SOC']-df['SOC_MIT'])
    df['PROC (NM)'] = (df['PROC']-df['PROC_MIT'])
    df['ENG (NM)'] = (df['ENG']-df['ENG_MIT'])
    df['WEA (NM)'] = (df['WEA']-df['WEA_MIT'])
    df['MGM (NM)'] = (df['MGM']-df['MGM_MIT'])
    return df

## UPDATES RISK EVENTS POST_MITIGATION IMPACTS

def update_impact (df, df_base, mitigation, df_coef):
    ''' This function updates the events partial impacts and it composition'''
    df['SOC_MIT'] =  (df_base['SOC_MIT']+(df_base['SOC']-df_base['SOC_MIT'])*(mitigation[0]))
    df['PROC_MIT'] = (df_base['PROC_MIT']+(df_base['PROC']-df_base['PROC_MIT'])*(mitigation[1]))
    df['ENG_MIT'] = (df_base['ENG_MIT']+(df_base['ENG']-df_base['ENG_MIT'])*(mitigation[2]))
    df['WEA_MIT'] = (df_base['WEA_MIT']+(df_base['WEA']-df_base['WEA_MIT'])*(mitigation[3]))
    df['MGM_MIT'] = (df_base['MGM_MIT']+(df_base['MGM']-df_base['MGM_MIT'])*(mitigation[4]))

    df['Social'] =       df_coef['SOC'] * df_base['SOC_EMEAN'] * (df['SOC']-df['SOC_MIT'] * df_coef['MIT_ef'])
    df['Procurement'] =  df_coef['PROC']* df_base['PROC_EMEAN']* (df['PROC']-df['PROC_MIT']*df_coef['MIT_ef'])
    df['Engineering'] =  df_coef['ENG'] * df_base['ENG_EMEAN'] * (df['ENG']-df['ENG_MIT'] * df_coef['MIT_ef'])
    df['Weather'] =      df_coef['WEA'] * df_base['WEA_EMEAN'] * (df['WEA']-df['WEA_MIT'] * df_coef['MIT_ef'])
    df['Management'] =   df_coef['MGM'] * df_base['MGM_EMEAN'] * (df['MGM']-df['MGM_MIT'] * df_coef['MIT_ef'])

    df['DEV_EVE'] = df['Social']+df['Procurement']+df['Engineering']+df['Weather']+df['Management']
    df['DEV_TOT'] = (1+df['DEV_EVE'])*(1+df['DEV_RAN'])-1
    return (df, mitigation)

def scatter_3dim (df, x_sel, y_sel, z_sel, size_by, color_by):
    fig_b = px.scatter_3d(df, x = x_sel, y = y_sel, z = z_sel, size = size_by, size_max = 20, color = color_by)
    return fig_b

## Complementary plot for correlation visualization
def scatter_hist (df, x_sel):
    fig_c = px.scatter(df, x = x_sel, y = 'DEV_EVE', color = 'DEV_TOT', marginal_y = 'box', marginal_x = 'box', width=600,
     title ='Non-mitigated correlation')
    return fig_c
